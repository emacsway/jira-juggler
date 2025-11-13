#! /usr/bin/python3
"""
Jira to task-juggler extraction script

This script queries Jira, and generates a task-juggler input file to generate a Gantt chart.
"""
import abc
import argparse
import copy
import csv
import functools
import logging
import math
import operator
import re
import typing
import businesstimedelta
from abc import ABC
import datetime
from functools import cmp_to_key
from getpass import getpass
from itertools import chain
from operator import attrgetter
from pathlib import Path

from dateutil import parser, tz
from decouple import config
from jira import JIRA, JIRAError
from natsort import natsorted, ns

DEFAULT_LOGLEVEL = 'warning'
DEFAULT_JIRA_URL = 'https://melexis.atlassian.net'
DEFAULT_OUTPUT = 'jira_export.tji'
TODO_STATUSES = (
    'to do',
    'blocked',
    'reopened',
    'postponed',
)
PROGRESS_STATUSES = (
    'in progress',
)
DEVELOPED_STATUSES = (
    'ready for code review',
    'in code review',
)
RESOLVED_STATUSES = (
    'approved',
    'resolved',
    'merged to dev'
)
PENDING_STATUSES = (
    'in testing',
    'ready for testing on qa',
    'ready for deployment',
)
DONE_STATUSES = (
    'closed',
    'cancelled',
)

JIRA_PAGE_SIZE = 50

TAB = ' ' * 2


def fetch_credentials():
    """ Fetches the credentials from the .env file by default or, alternatively, from the user's input

    Returns:
        str: email address or username
        str: API token or password
    """
    username = config('JIRA_USERNAME', default='')
    api_token = config('JIRA_API_TOKEN', default='')
    if not username:
        username = input('JIRA email address (or username): ')
    if not api_token:
        password = config('JIRA_PASSWORD', default='')
        if password:
            logging.warning('Basic authentication with a JIRA password may be deprecated. '
                            'Consider defining an API token as environment variable JIRA_API_TOKEN instead.')
            return username, password
        else:
            api_token = getpass(f'JIRA API token (or password) for {username}: ')
    return username, api_token


def set_logging_level(loglevel):
    """Sets the logging level

    Args:
        loglevel (str): String representation of the loglevel
    """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)


def to_identifier(key):
    """Converts given key to identifier, interpretable by TaskJuggler as a task-identifier

    Args:
        key (str): Key to be converted

    Returns:
        str: Valid task-identifier based on given key
    """
    return key.replace('-', '_')


def to_juggler_date(date):
    """Converts given datetime.datetime object to a string that can be interpreted by TaskJuggler

    The resolution is 60 minutes.

    Args:
        date (datetime.datetime): Datetime object

    Returns:
        str: String representing the date and time in TaskJuggler's format
    """
    return date.strftime('%Y-%m-%d-%H:00-%z').rstrip('-')


def calculate_weekends(date, workdays_passed, weeklymax):
    """Calculates the number of weekends between the given date and the amount of workdays to travel back in time.

    The following assumptions are made: each workday starts at 9 a.m., has no break and is 8 hours long.

    Args:
        date (datetime.datetime): Date and time specification to use as a starting point
        workdays_passed (float): Number of workdays passed since the given date
        weeklymax (int): Number of allocated workdays per week

    Returns:
        int: The number of weekends between the given date and the amount of weekdays that have passed since then
    """
    weekend_count = 0
    workday_percentage = (date - datetime.datetime.combine(date.date(), datetime.time(hour=9))).seconds / JugglerTaskEffort.FACTOR
    date_as_weekday = date.weekday() + workday_percentage
    if date_as_weekday > weeklymax:
        date_as_weekday = weeklymax
    remaining_workdays_passed = workdays_passed - date_as_weekday
    if remaining_workdays_passed > 0:
        weekend_count += 1 + (remaining_workdays_passed // weeklymax)
    return weekend_count


class AddWorkingDays:
    def __init__(self, weeklymax):
        self._workday = businesstimedelta.WorkDayRule(
            start_time=datetime.time(9),
            end_time=datetime.time(18),
            working_days=list(range(weeklymax)))

        # Take out the lunch break
        self._lunch_break = businesstimedelta.LunchTimeRule(
            start_time=datetime.time(12),
            end_time=datetime.time(13),
            working_days=list(range(weeklymax)))

        # Combine the two
        self._business_hrs = businesstimedelta.Rules([self._workday, self._lunch_break])

    def __call__(self, from_date, add_days):
        delta = businesstimedelta.BusinessTimeDelta(self._business_hrs, timedelta=datetime.timedelta(days=add_days))
        return from_date + delta


def to_username(value):
    """Converts the given value to a username (user ID), if needed, while caching the result.

    Args:
        value (str/jira.User): String (account ID or user ID) or User instance

    Returns:
        str: The corresponding username
    """
    user_id = value.accountId if hasattr(value, 'accountId') else str(value)
    if user_id in id_to_username_mapping:
        return id_to_username_mapping[user_id]

    if not isinstance(value, str):
        id_to_username_mapping[user_id] = determine_username(value)
    elif len(value) >= 24:  # accountId
        user = jirahandle.user(user_id)
        id_to_username_mapping[user_id] = determine_username(user)
    return id_to_username_mapping.get(user_id, value)


def determine_username(user):
    """Determines the username (user ID) for the given User.

    Args:
        user (jira.User): User instance

    Returns
        str: Corresponding username

    Raises:
        Exception: Failed to determine username
    """
    if getattr(user, 'emailAddress', ''):
        username = user.emailAddress.split('@')[0]
    elif getattr(user, 'name', ''):  # compatibility with Jira Server
        username = user.name
    elif getattr(user, 'displayName', ''):
        full_name = user.displayName
        username = f'"{full_name}"'
        logging.error(f"Failed to fetch email address of {full_name!r}: they restricted its visibility; "
                      f"using identifier {username!r} as fallback value.")
    else:
        raise Exception(f"Failed to determine username of {user}")
    return username


def determine_default_links(link_types_per_name):
    default_links = []
    for link_types in ({'Blocker': 'inward', 'Blocks': 'inward'}, {'Dependency': 'outward', 'Dependent': 'outward'}):
        for link_type_name, direction in link_types.items():
            if link_type_name in link_types_per_name:
                link = getattr(link_types_per_name[link_type_name], direction)
                default_links.append(link)
                break
        else:
            logging.warning("Failed to find any of these default jira-juggler issue link types in your Jira project "
                            f"configuration: {list(link_types)}. Use --links if you think this is a problem.")
    return default_links


def determine_links(jira_link_types, input_links):
    valid_links = set()
    if input_links is None:
        link_types_per_name = {link_type.name: link_type for link_type in jira_link_types}
        valid_links = determine_default_links(link_types_per_name)
    elif input_links:
        unique_input_links = set(input_links)
        all_jira_links = chain.from_iterable((link_type.inward, link_type.outward) for link_type in jira_link_types)
        missing_links = unique_input_links.difference(all_jira_links)
        if missing_links:
            logging.warning(f"Failed to find links {missing_links} in your configuration in Jira")
        valid_links = unique_input_links - missing_links
    return valid_links


class JugglerTaskProperty(ABC):
    """Class for a property of a Task Juggler"""

    DEFAULT_NAME = 'property name'
    DEFAULT_VALUE = 'not initialized'
    PREFIX = ''
    SUFFIX = ''
    TEMPLATE = TAB + '{prop} {value}\n'
    VALUE_TEMPLATE = '{prefix}{value}{suffix}'

    def __init__(self, jira_issue=None):
        """Initializes the task juggler property

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
            value (object): Value of the property
        """
        self.name = self.DEFAULT_NAME
        self.value = copy.copy(self.DEFAULT_VALUE)

        if jira_issue:
            self.load_from_jira_issue(jira_issue)

    @property
    def is_empty(self):
        """bool: True if the property contains an empty or uninitialized value"""
        return not self.value or self.value == self.DEFAULT_VALUE

    def clear(self):
        """Sets the name and value to the default"""
        self.name = self.DEFAULT_NAME
        self.value = self.DEFAULT_VALUE

    def load_from_jira_issue(self, jira_issue):
        """Loads the object with data from a Jira issue

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
        """

    def validate(self, task, tasks):
        """Validates (and corrects) the current task property

        Args:
            task (JugglerTask): Task to which the property belongs
            tasks (list): List of JugglerTask instances to which the current task belongs. Will be used to
                verify relations to other tasks.
        """

    def __str__(self):
        """Converts task property object to the task juggler syntax

        Returns:
            str: String representation of the task property in juggler syntax
        """
        if self.value is not None:
            return self.TEMPLATE.format(prop=self.name,
                                        value=self.VALUE_TEMPLATE.format(prefix=self.PREFIX,
                                                                         value=self.value,
                                                                         suffix=self.SUFFIX))
        return ''


class JugglerTaskAllocate(JugglerTaskProperty):
    """Class for the allocation (assignee) of a juggler task"""

    DEFAULT_NAME = 'allocate'
    DEFAULT_VALUE = '"not assigned"'

    def load_from_jira_issue(self, jira_issue):
        """Loads the object with data from a Jira issue.

        The last assignee in the Analyzed state of the Jira issue is prioritized over the current assignee,
        which is the fallback value.

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
        """
        if jira_issue.fields.status.name.lower() in DONE_STATUSES + PENDING_STATUSES + RESOLVED_STATUSES:
            before_resolved = False
            for change in sorted(jira_issue.changelog.histories, key=attrgetter('created'), reverse=True):
                for item in change.items:
                    if item.field.lower() == 'assignee':
                        if not before_resolved:
                            self.value = getattr(item, 'from', None)
                            if self.value:
                                self.value = to_username(self.value)
                        else:
                            self.value = to_username(item.to)
                            return  # got last assignee before transition to Approved/Resolved status
                    elif item.field.lower() == 'status' and item.toString.lower() in RESOLVED_STATUSES:
                        before_resolved = True
                        # if self.value and self.value != self.DEFAULT_VALUE:
                        #     return  # assignee was changed after transition to Closed/Resolved status

        if self.is_empty:
            if getattr(jira_issue.fields, 'assignee', None):
                self.value = to_username(jira_issue.fields.assignee)
            else:
                self.value = self.DEFAULT_VALUE

    def __str__(self):
        result = super().__str__().rstrip("\n")
        result += """ {\n%(tab)s%(tab)smandatory\n%(tab)s}\n""" % {'tab': TAB}
        return result


class JugglerTaskPriority(JugglerTaskProperty):
    """Class for the allocation (assignee) of a juggler task"""

    _PRIORITY_MAPPING = {
        'lowest': 200,
        'low': 350,
        'medium': 500,
        'high': 650,
        'highest': 800,
    }

    DEFAULT_NAME = 'priority'
    DEFAULT_VALUE = _PRIORITY_MAPPING['medium']

    def set_relatively_on(self, parent_priority=None):
        if parent_priority is not None:
            relative_priority = round((self.value - self._PRIORITY_MAPPING['medium']) * 0.03)
            if relative_priority != 0:
                self.value = parent_priority + relative_priority

    def load_from_jira_issue(self, jira_issue):
        if jira_issue.fields.priority:
            self.value = self._PRIORITY_MAPPING[jira_issue.fields.priority.name.lower()]

    def __str__(self):
        if self.value != self.DEFAULT_VALUE:
            return super().__str__()
        return ''


class IPertEstimate(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def optimistic(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nominal(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pessimistic(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def expected_duration(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def standard_deviation(self) -> float:
        raise NotImplementedError


class EmptyPertEstimate(IPertEstimate):

    @property
    def optimistic(self) -> float:
        return 0

    @property
    def nominal(self) -> float:
        return 0

    @property
    def pessimistic(self) -> float:
        return 0

    @property
    def expected_duration(self) -> float:
        return 0

    @property
    def standard_deviation(self) -> float:
        return 0


class PertEstimate:
    _optimistic: float
    _nominal: float
    _pessimistic: float

    def __init__(self, optimistic: float, nominal: float, pessimistic: float):
        assert optimistic is not None
        assert nominal is not None
        assert nominal >= optimistic
        assert pessimistic is not None
        assert pessimistic >= nominal

        self._optimistic = optimistic
        self._nominal = nominal
        self._pessimistic = pessimistic

    @property
    def optimistic(self) -> float:
        return self._optimistic

    @property
    def nominal(self) -> float:
        return self._nominal

    @property
    def pessimistic(self) -> float:
        return self._pessimistic

    @property
    def expected_duration(self) -> float:
        return (self.optimistic + 4*self.nominal + self.pessimistic) / 6

    @property
    def standard_deviation(self) -> float:
        return (self.pessimistic - self.optimistic) / 6


class CompositePertEstimate(IPertEstimate):
    _children: typing.Iterable[IPertEstimate]

    def __init__(self, children: typing.Iterable[IPertEstimate]):
        self._children = children

    @property
    def optimistic(self) -> float:
        return functools.reduce(
            operator.add,
            map(operator.attrgetter('optimistic'), self._children),
            0
        )

    @property
    def nominal(self) -> float:
        return functools.reduce(
            operator.add,
            map(operator.attrgetter('nominal'), self._children),
            0
        )

    @property
    def pessimistic(self) -> float:
        return functools.reduce(
            operator.add,
            map(operator.attrgetter('pessimistic'), self._children),
            0
        )

    @property
    def expected_duration(self) -> float:
        return functools.reduce(
            operator.add,
            map(operator.attrgetter('expected_duration'), self._children),
            0
        )

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(functools.reduce(
            operator.add,
            map(
                functools.partial(pow, exp=2),
                map(operator.attrgetter('expected_duration'), self._children)
            ),
            0
        ))


class JugglerTaskEffort(JugglerTaskProperty):
    """Class for the effort (estimate) of a juggler task"""

    # For converting the seconds (Jira) to days
    UNIT = 'd'
    FACTOR = 8.0 * 60 * 60

    DEFAULT_NAME = 'effort'
    MINIMAL_VALUE = 1.0 / 8
    DEFAULT_VALUE = MINIMAL_VALUE
    SUFFIX = UNIT
    pert: IPertEstimate = EmptyPertEstimate()

    def load_from_jira_issue(self, jira_issue):
        """Loads the object with data from a Jira issue

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
        """
        self.pert = jira_issue.pert
        if self.pert.expected_duration:
            self.value = round(self.pert.expected_duration, 3)
        elif hasattr(jira_issue.fields, 'timeoriginalestimate'):
            estimated_time = jira_issue.fields.timeoriginalestimate
            if estimated_time is not None:
                self.value = estimated_time / self.FACTOR
                logged_time = jira_issue.fields.timespent if jira_issue.fields.timespent else 0
                if jira_issue.fields.status.name.lower() in DONE_STATUSES + RESOLVED_STATUSES + PENDING_STATUSES:
                    # resolved ticket: prioritize Logged time over Estimated
                    if logged_time:
                        self.value = logged_time / self.FACTOR
                elif jira_issue.fields.timeestimate is not None:
                    # open ticket prioritize Remaining time over Estimated
                    if jira_issue.fields.timeestimate:
                        self.value = jira_issue.fields.timeestimate / self.FACTOR
                    else:
                        self.value = self.MINIMAL_VALUE
            else:
                self.value = self.DEFAULT_VALUE
        else:
            self.value = self.DEFAULT_VALUE
            logging.warning('No estimate found for %s, assuming %s%s', jira_issue.key, self.DEFAULT_VALUE, self.UNIT)

    def validate(self, task, tasks):
        """Validates (and corrects) the current task property

        Args:
            task (JugglerTask): Task to which the property belongs
            tasks (list): Modifiable list of JugglerTask instances to which the current task belongs. Will be used to
                verify relations to other tasks.
        """
        if self.value == 0:
            logging.warning('Estimate for %s, is 0. Excluding', task.key)
            tasks.remove(task)
        elif self.value < self.MINIMAL_VALUE:
            logging.warning('Estimate %s%s too low for %s, assuming %s%s', self.value, self.UNIT, task.key, self.MINIMAL_VALUE, self.UNIT)
            self.value = self.MINIMAL_VALUE

    def __str__(self):
        result = super().__str__()
        result += self.TEMPLATE.format(
            prop='stdev',
            value=self.VALUE_TEMPLATE.format(
                prefix='',
                value=round(self.pert.standard_deviation, 3),
                suffix='d'
            )
        )
        if not isinstance(self.pert, CompositePertEstimate):
            result += TAB + """${pert "%s" "%s" "%s"}\n""" % (
                self.pert.optimistic or self.MINIMAL_VALUE,
                self.pert.nominal or self.MINIMAL_VALUE,
                self.pert.pessimistic or self.MINIMAL_VALUE
            )
        return result


class Registry(dict):
    def path(self, key):
        if key not in self:
            return key
        path = []
        task = self[key]
        while task:
            path.append(to_identifier(task.key))
            task = task.parent
        path.reverse()
        return ".".join(path)


class JugglerTaskDepends(JugglerTaskProperty):
    """Class for linking of a juggler task"""

    DEFAULT_NAME = 'depends'
    DEFAULT_VALUE = []
    PREFIX = ''
    links = set()

    def __init__(self, registry, jira_issue=None):
        super().__init__(jira_issue)
        self.registry = registry

    def append_value(self, value):
        """Appends value for task juggler property

        Args:
            value (object): Value to append to the property
        """
        if value not in self.value:
            self.value.append(value)

    def load_from_jira_issue(self, jira_issue):
        """Loads the object with data from a Jira issue

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
        """
        if hasattr(jira_issue.fields, 'issuelinks'):
            for link in jira_issue.fields.issuelinks:
                if hasattr(link, 'inwardIssue') and link.type.inward in self.links:
                    self.append_value(to_identifier(link.inwardIssue.key))
                elif hasattr(link, 'outwardIssue') and link.type.outward in self.links:
                    self.append_value(to_identifier(link.outwardIssue.key))

    def validate(self, task, tasks):
        """Validates (and corrects) the current task property

        Args:
            task (JugglerTask): Task to which the property belongs
            tasks (list): List of JugglerTask instances to which the current task belongs. Will be used to
                verify relations to other tasks.
        """
        """
        task_ids = [self.registry.path(to_identifier(tsk.key)) for tsk in tasks]
        for val in list(self.value):
            if val not in task_ids:
                logging.warning('Removing link to %s for %s, as not within scope', val, task.key)
                self.value.remove(val)
        """

    def __str__(self):
        """Converts task property object to the task juggler syntax

        Returns:
            str: String representation of the task property in juggler syntax
        """
        if self.value:
            valstr = ''
            for val in self.value:
                val = self.registry.path(val)
                if valstr:
                    valstr += ', '
                valstr += self.VALUE_TEMPLATE.format(prefix=self.PREFIX,
                                                     value=val,
                                                     suffix=self.SUFFIX)
            return self.TEMPLATE.format(prop=self.name,
                                        value=valstr)
        return ''


class JugglerTaskFlags(JugglerTaskProperty):
    """Class for linking of a juggler task"""

    DEFAULT_NAME = 'flags'
    DEFAULT_VALUE = []

    def append_value(self, value):
        """Appends value for task juggler property

        Args:
            value (object): Value to append to the property
        """
        if value not in self.value:
            self.value.append(value)

    def load_from_jira_issue(self, jira_issue):
        """Loads the object with data from a Jira issue

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
        """
        pass

    def validate(self, task, tasks):
        pass

    def __str__(self):
        """Converts task property object to the task juggler syntax

        Returns:
            str: String representation of the task property in juggler syntax
        """
        if self.value:
            valstr = ''
            for val in self.value:
                if valstr:
                    valstr += ', '
                valstr += self.VALUE_TEMPLATE.format(prefix=self.PREFIX,
                                                     value=val,
                                                     suffix=self.SUFFIX)
            return self.TEMPLATE.format(prop=self.name,
                                        value=valstr)
        return ''


class JugglerTaskFactDepends(JugglerTaskDepends):
    DEFAULT_NAME = 'fact:depends'


class JugglerTaskTime(JugglerTaskProperty):
    """Class for setting the start/end time of a juggler task"""

    DEFAULT_VALUE = ''
    PREFIX = ''

    def load_from_jira_issue(self, jira_issue):
        start = self.do_get_start_date(jira_issue)
        fact_start = self.do_determine_fact_start_date(jira_issue)
        fact_end = self.do_determine_fact_end_date(jira_issue)
        logging.debug("""Date: %s %r %r""", jira_issue.key, start, fact_start, fact_end)
        if fact_end:
            if jira_issue.fields.status.name.lower() not in TODO_STATUSES:
                self.name, self.value = 'fact:end', fact_end
        elif fact_start:
            if jira_issue.fields.status.name.lower() not in TODO_STATUSES:
                self.name, self.value = 'fact:start', fact_start
        elif start:
            self.name, self.value = 'start', start
        logging.debug("""Set date: "%s", "%r", "%r""""", jira_issue.key, self.name, self.value)

    def do_get_start_date(self, issue):
        dt = getattr(issue.fields, 'customfield_10014', None)
        if dt:
            dt = datetime.datetime.strptime(dt, "%Y-%m-%d").date()
            return dt
        return None

    def do_determine_fact_start_date(self, issue):
        dt = None
        for change in sorted(issue.changelog.histories, key=attrgetter('created'), reverse=True):
            for item in change.items:
                if item.field.lower() == 'status':
                    status = item.toString.lower()
                    if status in PROGRESS_STATUSES:
                        return parser.isoparse(change.created)
                    elif status in DEVELOPED_STATUSES and dt is None:
                        dt = parser.isoparse(change.created)
        return dt

    def do_determine_fact_end_date(self, issue):
        dt = None
        for change in sorted(issue.changelog.histories, key=attrgetter('created'), reverse=True):
            for item in change.items:
                if item.field.lower() == 'status':
                    status = item.toString.lower()
                    if status in RESOLVED_STATUSES:
                        return parser.isoparse(change.created)
                    elif status in DONE_STATUSES + PENDING_STATUSES and dt is None:
                        dt = parser.isoparse(change.created)
        return dt

    def validate(self, *_):
        """Validates the current task property"""
        if not self.is_empty:
            valid_names = ('start', 'fact:start', 'end', 'fact:end',)
            if self.name not in valid_names:
                raise ValueError(f'The name of {self.__class__.__name__} is invalid; expected a value in {valid_names}')

    def __str__(self):
        """Converts task property object to the task juggler syntax

        Returns:
            str: String representation of the task property in juggler syntax
        """
        if self.value:
            return self.TEMPLATE.format(
                prop=self.name,
                value=to_juggler_date(self.value)
            )
        return ''


class JugglerTaskComplete(JugglerTaskProperty):
    DEFAULT_NAME = 'complete'
    DEFAULT_VALUE = 0

    def load_from_jira_issue(self, jira_issue):
        progress = getattr(jira_issue.fields, 'progress', None)
        if progress and progress.progress and progress.total:
            self.value = math.ceil(100 * progress.progress / progress.total)
        elif jira_issue.fields.status.name.lower() in ('in progress', 'reopened'):
            self.value = 50
        elif jira_issue.fields.status.name.lower() in DEVELOPED_STATUSES:
            self.value = 80
        elif jira_issue.fields.status.name.lower() in DONE_STATUSES + RESOLVED_STATUSES + PENDING_STATUSES:
            self.value = 100
        if self.value == 0:
            self.value = self.extract_status_from_history(jira_issue)

    def extract_status_from_history(self, issue):
        for change in sorted(issue.changelog.histories, key=attrgetter('created'), reverse=True):
            for item in change.items:
                if item.field.lower() == 'status':
                    status = item.toString.lower()
                    if status in RESOLVED_STATUSES + DEVELOPED_STATUSES:
                        return 80
                    elif status in PROGRESS_STATUSES:
                        return 50
        return 0


class JugglerTask:
    """Class for a task for Task-Juggler"""

    class TYPE:
        EPIC = 'Epic'
        SPIKE = 'Spike'
        STORY = 'Story'
        SUBTASK = 'Sub-task'
        IMPROVEMENT = 'Improvement'
        FEATURE = 'New Feature'

    add_working_days: AddWorkingDays
    focus_factor = 0.7
    children: list['JugglerTask']
    DEFAULT_KEY = 'NOT_INITIALIZED'
    MAX_SUMMARY_LENGTH = 70
    DEFAULT_SUMMARY = 'Task is not initialized'
    TEMPLATE = '''
task {id} "{description}" {{
{tab}Key "{key}"
{props}
{children}
}}
'''

    @classmethod
    def factory(cls, registry, jira_issue, parent=None) -> 'JugglerTask':
        if jira_issue.fields.issuetype.name == cls.TYPE.EPIC:
            return Epic(registry, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.SPIKE or '[spike]' in jira_issue.fields.summary.lower():
            return Spike(registry, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.SUBTASK and "[qa auto]" in jira_issue.fields.summary.lower():
            return QaAutoSubtask(registry, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.SUBTASK and "[qa manual]" in jira_issue.fields.summary.lower():
            return QaManualSubtask(registry, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.SUBTASK:
            return Subtask(registry, jira_issue, parent)
        else:
            return BacklogItem(registry, jira_issue, parent)

    def __init__(self, registry, jira_issue=None, parent=None):
        logging.info('Create JugglerTask for %s', jira_issue.key)

        self.key = self.DEFAULT_KEY
        self.summary = self.DEFAULT_SUMMARY
        self.properties = {}
        self.issue = None
        self.type = None
        self._resolved_at_date = None
        self.parent = parent
        self.registry = registry

        if jira_issue:
            self.load_from_jira_issue(jira_issue)

    def _inherit_priority(self):
        if self.parent.properties['priority'].value > self.properties['priority'].value:
            self.properties['priority'].value = self.parent.properties['priority'].value

    def load_from_jira_issue(self, jira_issue):
        """Loads the object with data from a Jira issue

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
        """
        self.key = jira_issue.key
        self.issue = jira_issue
        self.type = jira_issue.fields.issuetype.name
        summary = jira_issue.fields.summary.replace('\"', '\\\"')
        self.summary = (summary[:self.MAX_SUMMARY_LENGTH] + '...') if len(summary) > self.MAX_SUMMARY_LENGTH else summary
        if self.is_resolved:
            self.resolved_at_date = self.determine_resolved_at_date()
        self.properties['allocate'] = JugglerTaskAllocate(jira_issue)
        self.properties['effort'] = JugglerTaskEffort(jira_issue)
        self.properties['depends'] = JugglerTaskDepends(self.registry, jira_issue)
        self.properties['fact:depends'] = JugglerTaskFactDepends(self.registry)
        self.properties['time'] = JugglerTaskTime(jira_issue)
        self.properties['complete'] = JugglerTaskComplete(jira_issue)
        self.properties['priority'] = JugglerTaskPriority(jira_issue)
        self.properties['flags'] = JugglerTaskFlags(jira_issue)
        self.children = [JugglerTask.factory(self.registry, child, self) for child in jira_issue.children]
        self.registry[to_identifier(self.key)] = self

    def validate(self, tasks, property_identifier):
        """Validates (and corrects) the current task

        Args:
            tasks (list): List of JugglerTask instances to which the current task belongs. Will be used to
                verify relations to other tasks.
            property_identifier (str): Identifier of property type
        """
        if self.key == self.DEFAULT_KEY:
            logging.error('Found a task which is not initialized')
        self.properties[property_identifier].validate(self, tasks)

        for task in tasks:
            if task.children:
                self.validate(task.children, property_identifier)

    def __str__(self):
        """Converts the JugglerTask to the task juggler syntax

        Returns:
            str: String representation of the task in juggler syntax
        """
        props = list()
        for k, v in self.properties.items():
            if k in ('effort', 'allocate', 'complete',) and self.children:
                continue
            props.append(str(v))

        props = "".join(props)
        result = self.TEMPLATE.format(
            id=to_identifier(self.key),
            key=self.key,
            tab=TAB,
            description=self.summary.replace('\"', '\\\"'),
            props=props,
            children="".join(map(self._make_indentation, self.children))
        )
        return result

    @staticmethod
    def _make_indentation(task):
        result = []
        for line in str(task).split("\n"):
            if len(line):
                line = TAB + line
            result.append(line)
        return "\n".join(result)

    @property
    def is_resolved(self):
        """bool: True if JIRA issue has been approved/resolved/closed; False otherwise"""
        return self.issue is not None and self.issue.fields.status.name.lower() in (
                DONE_STATUSES + RESOLVED_STATUSES + PENDING_STATUSES
        )

    @property
    def resolved_at_date(self):
        """datetime.datetime: Date and time corresponding to the last transition to the Approved/Resolved status; the
            transition to the Closed status is used as fallback; None when not resolved
        """
        return self._resolved_at_date

    @resolved_at_date.setter
    def resolved_at_date(self, value):
        self._resolved_at_date = value

    def determine_resolved_at_date(self):
        closed_at_date = None
        for change in sorted(self.issue.changelog.histories, key=attrgetter('created'), reverse=True):
            for item in change.items:
                if item.field.lower() == 'status':
                    status = item.toString.lower()
                    if status in RESOLVED_STATUSES:
                        return parser.isoparse(change.created)
                    elif status in DONE_STATUSES + PENDING_STATUSES and closed_at_date is None:
                        closed_at_date = parser.isoparse(change.created)
        return closed_at_date

    def shift_unstarted_tasks_to_milestone(self, extras, milestone):
        sprint = self.get_sprint(extras)
        if sprint:
            self.properties['fact:depends'].append_value(sprint)
            if not self.time_is_empty() and self.in_progress():
                for child in self.children:
                    if child.time_is_empty():
                        child.shift_unstarted_tasks_to_milestone(extras, milestone)
        elif self.time_is_empty():
            self.properties['fact:depends'].append_value(milestone)
        elif self.children:
            for child in self.children:
                child.shift_unstarted_tasks_to_milestone(extras, milestone)

    def get_sprint(self, sprint_backlogs):
        sprint = None
        if self.key in sprint_backlogs:
            sprint = sprint_backlogs[self.key].sprint
        if not sprint and getattr(self, 'sprint', None):
            sprint = getattr(self, 'sprint').name
        return sprint

    def adjust_priority(self, extras):
        priority = None
        if self.key in extras:
            priority = extras[self.key].priority
        if priority is not None:
            self.properties['priority'].value = priority
        if self.children:
            for child in self.children:
                child.adjust_priority(extras)

    def is_dor(self):
        if self.children:
            return any(child.is_dor() for child in self.children)  # all()?
        else:
            return not self.properties['effort'].is_empty

    def todo(self):
        if self.children:
            return all(child.todo() for child in self.children)
        else:
            return self.properties['complete'].is_empty

    def in_progress(self):
        if self.children:
            if any(child.in_progress() for child in self.children):
                return True
            complete = any(child.is_complete() for child in self.children)
            todo = any(child.todo() for child in self.children)
            return complete and todo
        else:
            return 0 < self.properties['complete'].value < 100

    def is_complete(self):
        if self.children:
            return all(child.is_complete() for child in self.children)
        else:
            return self.properties['complete'].value == 100

    def adjust_flags(self, extras):
        if self.key in extras:
            flags = extras[self.key].flags
            for flag in flags:
                self.properties['flags'].append_value(flag)
        if self.children:
            for child in self.children:
                child.adjust_flags(extras)

    def collect_todo_tasks(self, collector: Registry):
        if self.time_is_empty():
            collector[to_identifier(self.key)] = self
        elif self.children:
            for child in self.children:
                child.collect_todo_tasks(collector)

    def bottom_up_deps(self) -> set['JugglerTask']:
        deps = set()  # TODO: Use OrderedSet?
        for dep in self.properties['depends'].value:
            if dep in self.properties['depends'].registry:
                deps.add(self.properties['depends'].registry.get(dep))
        if self.parent:
            deps = deps.union(self.parent.bottom_up_deps())
        for dep in deps:
            deps = deps.union(dep.bottom_up_deps())
        return deps

    def time_is_empty(self) -> bool:
        if not self.properties['time'].is_empty:
            return False
        for child in self.children:
            if not child.time_is_empty():
                return False
        return True

    def max_time(self) -> datetime.datetime | None:
        dts = []
        if not self.properties['time'].is_empty:
            if 'end' in self.properties['time'].name:
                end_time = self.properties['time'].value
            else:
                start_time = self.properties['time'].value
                working_days = self.properties['effort'].value / self.focus_factor
                end_time = self.add_working_days(start_time, working_days)
            dts.append(end_time)
        for child in self.children:
            dt = child.max_time()
            if dt:
                dts.append(dt)
        if dts:
            return max(dts)
        return None

    def fix_time(self):
        if self.children:
            for child in self.children:
                child.fix_time()
        for dep in self.bottom_up_deps():
            if isinstance(dep, Spike):
                if dep.is_resolved:
                    dt = dep.max_time()
                    if dt is not None:
                        if not self.properties['time'].is_empty:
                            if 'start' in self.properties['time'].name:
                                start_time = self.properties['time'].value
                            else:
                                end_time = self.properties['time'].value
                                days_spent = self.properties['effort'].value / self.focus_factor
                                start_time = self.add_working_days(end_time, -days_spent)
                            if dt > start_time:
                                logging.warning(
                                    """Fix time for %s "%s" %s because of resolved dependency %s "%s" %s""",
                                    self.key, self.summary, self.properties['time'],
                                    dep.key, dep.summary, dt
                                )
                                self.properties['time'] = JugglerTaskTime()
                elif dep.time_is_empty():
                    if not self.properties['time'].is_empty:
                        logging.warning(
                            """Fix time for %s "%s" %s because of unresolved dependency %s "%s" with no time""",
                            self.key, self.summary, self.properties['time'],
                            dep.key, dep.summary
                        )
                        self.properties['time'] = JugglerTaskTime()

    def shift_in_progress_to(self, current_date):
        if self.children:
            for child in self.children:
                child.shift_in_progress_to(current_date)
        elif not self.time_is_empty():
            progress = self.properties['complete'].value
            if 0 < progress < 100:
                self.properties['time'].name = 'fact:start'
                days_spent = self.properties['effort'].value / self.focus_factor
                self.properties['time'].value = self.add_working_days(current_date, -days_spent)

    def sort(self, *, key=None, reverse=False):
        if self.children:
            self.children.sort(key=key, reverse=reverse)
            for child in self.children:
                child.sort(key=key, reverse=reverse)


class Epic(JugglerTask):

    def shift_unstarted_tasks_to_milestone(self, extras, milestone):
        for child in self.children:
            child.shift_unstarted_tasks_to_milestone(extras, milestone)

    def collect_todo_tasks(self, collector: Registry):
        for child in self.children:
            child.collect_todo_tasks(collector)


class BacklogItem(JugglerTask):

    def load_from_jira_issue(self, jira_issue):
        super().load_from_jira_issue(jira_issue)

        manual_testing_tasks = [child for child in self.children if isinstance(child, QaManualSubtask)]
        if manual_testing_tasks:
            manual_testing_dependencies = [
                child for child in self.children if not isinstance(child, (QaAutoSubtask, QaManualSubtask))
            ]
            for manual_testing_task in manual_testing_tasks:
                if manual_testing_task.time_is_empty():
                    for manual_testing_dependency in manual_testing_dependencies:
                        manual_testing_task.properties['depends'].append_value(
                            to_identifier(manual_testing_dependency.key)
                        )

    def shift_unstarted_tasks_to_milestone(self, extras, milestone):
        if not self.is_dor():
            milestone = "${sprint_non_dor}"
        super().shift_unstarted_tasks_to_milestone(extras, milestone)

    def adjust_priority(self, extras):
        priority = None
        if self.key in extras:
            priority = extras[self.key].priority
        if priority is not None:
            self.properties['priority'].value = priority
            if self.children:
                for child in self.children:
                    child.adjust_priority(extras)
        elif not self.is_dor() and self.todo():
            self.properties['priority'].value = 1
        elif self.children:
            for child in self.children:
                child.adjust_priority(extras)


class Spike(JugglerTask):
    pass


class Subtask(JugglerTask):
    def load_from_jira_issue(self, jira_issue):
        super().load_from_jira_issue(jira_issue)
        # self._inherit_priority()
        # self.properties['priority'].clear()
        self.properties['priority'].set_relatively_on(self.parent.properties['priority'].value)

    def adjust_priority(self, extras):
        super().adjust_priority(extras)
        self.properties['priority'].set_relatively_on(self.parent.properties['priority'].value)


class QaAutoSubtask(Subtask):
    pass


class QaManualSubtask(Subtask):
    pass


class JiraJuggler:
    """Class for task-juggling Jira results"""

    def __init__(self, endpoint, user, token, query, kids_query,  links=None):
        """Constructs a JIRA juggler object

        Args:
            endpoint (str): Endpoint for the Jira Cloud (or Server)
            user (str): Email address (or username)
            token (str): API token (or password)
            query (str): The query to run
            links (set/None): List of issue link type inward/outward links; None to use the default configuration
        """
        global id_to_username_mapping
        id_to_username_mapping = {}
        logging.info('Jira endpoint: %s', endpoint)

        global jirahandle
        jirahandle = JIRA(endpoint, basic_auth=(user, token), options={'rest_api_version': 3})
        if 'ORDER BY' not in query.upper():
            query = "%s ORDER BY priority DESC, created ASC" % query
        logging.info('Query: %s', query)
        self.query = query
        self.kids_query = kids_query

        all_jira_link_types = jirahandle.issue_link_types()
        JugglerTaskDepends.links = determine_links(all_jira_link_types, links)

    @classmethod
    def validate_tasks(cls, tasks):
        """Validates (and corrects) tasks

        Args:
            tasks (list): List of JugglerTask instances to validate
        """
        for property_identifier in ('allocate', 'effort', 'depends', 'time'):
            for task in list(tasks):
                task.validate(tasks, property_identifier)

    def load_issues_from_jira(self, depend_on_preceding=False, sprint_field_name='', **kwargs):
        """Loads issues from Jira

        Args:
            depend_on_preceding (bool): True to let each task depend on the preceding task that has the same user
                allocated to it, unless it is already linked; False to not add these links
            sprint_field_name (str): Name of field to sort tasks on

        Returns:
            list: A list of JugglerTask instances
        """
        registry = Registry()
        tasks = [JugglerTask.factory(registry, issue) for issue in self._load_issues(self.query)]
        self.validate_tasks(tasks)
        if sprint_field_name:
            self.sort_tasks_on_sprint(tasks, sprint_field_name)
        tasks.sort(key=cmp_to_key(self.compare_status))
        if depend_on_preceding:
            self.link_to_preceding_task(tasks, **kwargs)
        return tasks

    def _load_issues(self, query):
        next_page_token = None
        result = []
        while True:
            try:
                response = jirahandle.enhanced_search_issues(
                    query,
                    maxResults=JIRA_PAGE_SIZE,
                    expand='changelog',
                    nextPageToken=next_page_token
                )
            except JIRAError as err:
                logging.error(f'Failed to query JIRA: {err}')
                if err.status_code == 401:
                    logging.error('Please check your JIRA credentials in the .env file or environment variables.')
                elif err.status_code == 403:
                    logging.error('You do not have permission to access this JIRA project or query.')
                elif err.status_code == 404:
                    logging.error('The JIRA endpoint is not found. Please check the endpoint URL.')
                elif err.status_code == 400:
                    # Parse and display the specific JQL errors more clearly
                    try:
                        error_data = err.response.json()
                        if 'errorMessages' in error_data:
                            for error_msg in error_data['errorMessages']:
                                logging.error(f'JIRA query error: {error_msg}')
                    except Exception:
                        pass  # Fall back to generic error if JSON parsing fails

                    logging.error('Invalid JQL query syntax. Please check your query.')
                else:
                    logging.error(f'An unexpected error occurred: {err}')
                return None

            if hasattr(response, 'issues'):
                issues = response.issues
            elif isinstance(response, dict) and 'issues' in response:
                issues = response['issues']
            else:
                issues = response

            for issue in issues:
                logging.debug(f'Retrieved {issue.key}: {issue.fields.summary}')
                if issue.fields.status.name.lower() == "closed" and getattr(issue.fields.resolution, 'name', None) == "Won't Do":
                    continue
                elif issue.fields.status.name.lower() == "cancelled":
                    continue
                result.append(issue)

            if hasattr(response, 'nextPageToken'):
                next_page_token = response.nextPageToken
            elif isinstance(response, dict) and 'nextPageToken' in response:
                next_page_token = response['nextPageToken']
            else:
                next_page_token = None
            logging.debug("Next page token: %s" % next_page_token)
            if not next_page_token:
                break

        for issue in result:
            self._attach_children(issue)
            self._attach_pert_estimate(issue)

        return result

    def _attach_children(self, issue):
        extra_query = "AND (%s)" % self.kids_query if self.kids_query else ""
        issue.children = self._load_issues(
            """parent = %s %s ORDER BY priority DESC, created ASC""" % (issue.key, extra_query)
        )

    def _attach_pert_estimate(self, issue):
        if len(issue.children) > 0:
            issue.pert = CompositePertEstimate(
                [i.pert for i in issue.children]
            )
        else:
            issue.pert = self.do_get_pert_estimate(issue)

    def do_get_pert_estimate(self, issue):
        try:
            pert_response = jirahandle.issue_property(issue.key, 'pert-estimation')
        except JIRAError as e:
            if e.status_code == 404:
                return EmptyPertEstimate()
            else:
                raise
        else:
            return PertEstimate(
                optimistic=pert_response.value.optimistic,
                nominal=pert_response.value.nominal,
                pessimistic=pert_response.value.pessimistic,
            )

    def juggle(self, output=None, **kwargs):
        """Queries JIRA and generates task-juggler output from given issues

        Args:
            list: A list of JugglerTask instances
        """
        JugglerTask.add_working_days = staticmethod(AddWorkingDays(kwargs.get('weeklymax')))
        if kwargs.get('sprint_field_name'):
            kwargs['sprint_field_name'], sprint_re_pattern, sprint_re_repl = kwargs['sprint_field_name'].split('|')
            JugglerTask.sprint_accessor = staticmethod(make_sprint_accessor(
                kwargs['sprint_field_name'],
                sprint_re_pattern,
                sprint_re_repl
            ))
        juggler_tasks = self.load_issues_from_jira(**kwargs)
        if not juggler_tasks:
            return None

        extras = {}
        if kwargs.get('extras_filepath'):
            extras = self._load_extras(kwargs['extras_filepath'])

        for task in juggler_tasks:
            task.adjust_flags(extras)
            task.adjust_priority(extras)

        juggler_tasks.sort(key=lambda i: i.properties['priority'].value, reverse=True)
        for task in juggler_tasks:
            task.sort(key=lambda i: i.properties['priority'].value, reverse=True)

        # for task in juggler_tasks:
        #     task.shift_in_progress_to(kwargs['current_date'])

        if kwargs.get('milestone'):
            for task in juggler_tasks:
                task.fix_time()

            collector = Registry()
            for task in juggler_tasks:
                task.shift_unstarted_tasks_to_milestone(extras, kwargs['milestone'])
                # task.collect_todo_tasks(collector)
            if False and output:
                path = Path(output)
                new_name = path.parent / ("%s_sprints%s.tmpl" % (path.stem, path.suffix))
                # new_name = path.with_suffix('').with_name(path.stem + "_sprints").with_suffix(path.suffix + ".tmpl")
                with open(new_name, 'w', encoding='utf-8') as out:
                    for k, task in collector.items():
                        out.write("""
supplement task %(id)s {
    fact:depends %(sprint)s
}
""" % {'id': collector.path(k), 'sprint': kwargs['milestone']})
        if output:
            with open(output, 'w', encoding='utf-8') as out:
                for task in juggler_tasks:
                    out.write(str(task))
        return juggler_tasks

    @staticmethod
    def _load_extras(filepath):
        extras = {}
        with open(filepath, newline='') as csvfile:
            for row in csv.reader(csvfile):
                if not row or row[0].startswith('#'):
                    continue
                extras[row[0]] = TaskExtra(
                    sprint=row[1] or None,
                    priority=int(row[2]) if len(row) > 2 and row[2] else None,
                    flags=[i.strip() for i in row[3].split(',')] if row[3] else [],
                )
        return extras

    @staticmethod
    def link_to_preceding_task(tasks, weeklymax=5, current_date=datetime.datetime.now(tz.tzutc()), **kwargs):
        """Links task to preceding task with the same assignee.

        If the task has been resolved, 'end' is added instead of 'depends' no matter what, followed by the
        date and time on which it's been resolved.

        If it's the first unresolved task for a given assignee, 'start' is added followed by the date and hour on which
        the task has been started, i.e. current time minus time spent.
        For the other unresolved tasks, the effort estimate is 'Remaining' time
        only instead of 'Remaining + Logged' time since parallellism is not supported by
        TaskJuggler and this approach results in a more accurate forecast.

        Args:
            tasks (list): List of JugglerTask instances to modify
            weeklymax (int): Number of allocated workdays per week
            current_date (datetime.datetime): Offset-naive datetime.datetime to treat as the current date
        """
        id_to_task_map = {to_identifier(task.key): task for task in tasks}
        unresolved_tasks = {}
        for task in tasks:
            assignee = str(task.properties['allocate'])

            depends_property = task.properties['depends']
            time_property = task.properties['time']
            logging.debug('Before %s %s %s', task.key, time_property.name, time_property.value)
            if task.is_resolved:
                depends_property.clear()  # don't output any links from JIRA
                time_property.name = 'end'
                time_property.value = task.resolved_at_date
            else:
                if assignee in unresolved_tasks:  # link to a preceding unresolved task
                    preceding_task = unresolved_tasks[assignee][-1]
                    depends_property.append_value(to_identifier(preceding_task.key))
                else:  # first unresolved task for assignee: set start time unless it depends on an unresolved task
                    for identifier in depends_property.value:
                        if not id_to_task_map[identifier].is_resolved:
                            break
                    else:
                        start_time = current_date
                        if task.issue.fields.timespent:
                            effort_property = task.properties['effort']
                            effort_property.value += task.issue.fields.timespent / JugglerTaskEffort.FACTOR
                            days_spent = task.issue.fields.timespent // 3600 / 8
                            weekends = calculate_weekends(current_date, days_spent, weeklymax)
                            days_per_weekend = min(2, 7 - weeklymax)
                            start_time = current_date - datetime.datetime(days=(days_spent + weekends * days_per_weekend))
                        time_property.name = 'start'
                        time_property.value = start_time

                unresolved_tasks.setdefault(assignee, []).append(task)
            logging.debug('After %s %s %s', task.key, time_property.name, time_property.value)

    def sort_tasks_on_sprint(self, tasks, sprint_field_name):
        """Sorts given list of tasks based on the values of the field with the given name.

        JIRA issues that are not assigned to a sprint will be ordered last.

        Args:
            tasks (list): List of JugglerTask instances to sort in place
            sprint_field_name (str): Name of the field that contains information about sprints
        """
        priorities = {
            "ACTIVE": 3,
            "FUTURE": 2,
            "CLOSED": 1,
        }
        for task in tasks:
            task.sprint_name = ""
            task.sprint_priority = 0
            task.sprint_start_date = None
            if not task.issue:
                continue
            values = getattr(task.issue.fields, sprint_field_name, None)
            if values is not None:
                if isinstance(values, str):
                    values = [values]
                for sprint_info in values:
                    state = ""
                    if isinstance(sprint_info, (str, bytes)):  # Jira Server
                        state_match = re.search("state=({})".format("|".join(priorities)), sprint_info)
                        if state_match:
                            state = state_match.group(1)
                            prio = priorities[state]
                            if prio > task.sprint_priority:
                                task.sprint_name = re.search("name=(.+?),", sprint_info).group(1)
                                task.sprint_priority = prio
                                task.sprint_start_date = self.extract_start_date(sprint_info, task.issue.key)
                    else:  # Jira Cloud
                        state = sprint_info.state.upper()
                        if state in priorities:
                            prio = priorities[state]
                            if prio > task.sprint_priority:
                                task.sprint_name = sprint_info.name
                                task.sprint_priority = prio
                                if hasattr(sprint_info, 'startDate'):
                                    task.sprint_start_date = parser.parse(sprint_info.startDate)
                if task.children:
                    self.sort_tasks_on_sprint(task.children, sprint_field_name)

        logging.debug("Sorting tasks based on sprint information...")
        tasks.sort(key=cmp_to_key(self.compare_sprint_priority))

    @staticmethod
    def extract_start_date(sprint_info, issue_key):
        """Extracts the start date from the given info string.

        Args:
            sprint_info (str): Raw information about a sprint, as returned by the JIRA API
            issue_key (str): Name of the JIRA issue

        Returns:
            datetime.datetime/None: Start date as a datetime.datetime object or None if the sprint does not have a start date
        """
        start_date_match = re.search("startDate=(.+?),", sprint_info)
        if start_date_match:
            start_date_str = start_date_match.group(1)
            if start_date_str != '<null>':
                try:
                    return parser.parse(start_date_match.group(1))
                except parser.ParserError as err:
                    logging.debug("Failed to parse start date of sprint of issue %s: %s", issue_key, err)
                    return None

    @staticmethod
    def compare_sprint_priority(a, b):
        """Compares the priority of two tasks based on the sprint information

        The sprint_priority attribute is taken into account first, followed by the sprint_start_date and, lastly, the
        sprint_name attribute using natural sorting (a sprint with the word 'backlog' in its name is sorted as last).

        Args:
            a (JugglerTask): First JugglerTask instance in the comparison
            b (JugglerTask): Second JugglerTask instance in the comparison

        Returns:
            int: 0 for equal priority; -1 to prioritize a over b; 1 otherwise
        """
        if a.sprint_priority > b.sprint_priority:
            return -1
        if a.sprint_priority < b.sprint_priority:
            return 1
        if a.sprint_priority == 0 or a.sprint_name == b.sprint_name:
            return 0  # no/same sprint associated with both issues
        if type(a.sprint_start_date) != type(b.sprint_start_date):  # noqa
            return -1 if b.sprint_start_date is None else 1
        if a.sprint_start_date == b.sprint_start_date:
            # a sprint with backlog in its name has lower priority
            if "backlog" not in a.sprint_name.lower() and "backlog" in b.sprint_name.lower():
                return -1
            if "backlog" in a.sprint_name.lower() and "backlog" not in b.sprint_name.lower():
                return 1
            if natsorted([a.sprint_name, b.sprint_name], alg=ns.IGNORECASE)[0] == a.sprint_name:
                return -1
            return 1
        if a.sprint_start_date < b.sprint_start_date:
            return -1
        return 1

    @staticmethod
    def compare_status(a, b):
        if a.is_resolved and not b.is_resolved:
            return -1
        if b.is_resolved and not a.is_resolved:
            return 1
        if a.is_resolved and b.is_resolved:
            if a.resolved_at_date < b.resolved_at_date:
                return -1
            return 1
        return 0


class TaskExtra:
    def __init__(self, sprint, priority, flags):
        self.sprint = sprint
        self.priority = priority
        self.flags = flags


class Sprint:
    def __init__(self, name, priority, start):
        self.name = name
        self.priority = priority
        self.start = start


def make_sprint_accessor(sprint_field_name, sprint_re_pattern, sprint_re_repl):
    pattern = re.compile(sprint_re_pattern)
    priorities = {
        "ACTIVE": 3,
        "FUTURE": 2,
        "CLOSED": 1,
    }

    def sprint_accessor(jira_issue):
        sprint = Sprint("", 0, None)
        values = getattr(jira_issue.fields, sprint_field_name, None)
        if values is not None:
            if isinstance(values, str):
                values = [values]
            for sprint_info in values:
                state = ""
                if isinstance(sprint_info, (str, bytes)):  # Jira Server
                    state_match = re.search("state=({})".format("|".join(priorities)), sprint_info)
                    if state_match:
                        state = state_match.group(1)
                        prio = priorities[state]
                        if prio > sprint.priority:
                            name = re.search("name=(.+?),", sprint_info).group(1)
                            name = pattern.sub(sprint_re_repl, name)
                            sprint = Sprint(
                                name,
                                prio,
                                extract_start_date(sprint_info, jira_issue.key)
                            )

                else:  # Jira Cloud
                    state = sprint_info.state.upper()
                    if state in priorities:
                        prio = priorities[state]
                        if prio > sprint.priority:
                            name = sprint_info.name
                            name = pattern.sub(sprint_re_repl, name)
                            sprint = Sprint(
                                name,
                                prio,
                                parser.parse(sprint_info.startDate) if hasattr(sprint_info, 'startDate') else None
                            )
        return sprint

    return sprint_accessor


def extract_start_date(sprint_info, issue_key):
    """Extracts the start date from the given info string.

    Args:
        sprint_info (str): Raw information about a sprint, as returned by the JIRA API
        issue_key (str): Name of the JIRA issue

    Returns:
        datetime.datetime/None: Start date as a datetime.datetime object or None if the sprint does not have a start date
    """
    start_date_match = re.search("startDate=(.+?),", sprint_info)
    if start_date_match:
        start_date_str = start_date_match.group(1)
        if start_date_str != '<null>':
            try:
                return parser.parse(start_date_match.group(1))
            except parser.ParserError as err:
                logging.debug("Failed to parse start date of sprint of issue %s: %s", issue_key, err)
                return None


def main():
    argpar = argparse.ArgumentParser()
    argpar.add_argument('-l', '--loglevel', default=DEFAULT_LOGLEVEL,
                        help='Level for logging (strings from logging python package)')
    argpar.add_argument('-q', '--query', required=True,
                        help='Query to perform on JIRA server')
    argpar.add_argument('-k', '--kids-query', required=False,
                        help='Kids query to perform on JIRA server')
    argpar.add_argument('-o', '--output', default=DEFAULT_OUTPUT,
                        help='Output .tjp or .tji file for task-juggler')
    argpar.add_argument('-L', '--links', nargs='*',
                        help="Specific issue link type inward/outward links to consider for TaskJuggler's 'depends' "
                        "keyword, e.g. 'depends on'. "
                        "By default, link types Dependency/Dependent (outward only) and Blocker/Blocks (inwardy only) "
                        "are considered.  Specify an empty value to ignore Jira issue links altogether.")
    argpar.add_argument('-D', '--depend-on-preceding', action='store_true',
                        help='Flag to let tasks depend on the preceding task with the same assignee')
    argpar.add_argument('-s', '--sort-on-sprint', dest='sprint_field_name', default='',
                        help='Sort unresolved tasks by using field name that stores sprint(s), e.g. customfield_10851, '
                             'in addition to the original order')
    argpar.add_argument('-w', '--weeklymax', default=5, type=int,
                        help='Number of allocated workdays per week used to approximate '
                             'start time of unresolved tasks with logged time')
    argpar.add_argument('-c', '--current-date', default=datetime.datetime.now(tz.tzutc()), type=parser.isoparse,
                        help='Specify the offset-naive date to use for calculation as current date. If no value is '
                             'specified, the current value of the system clock is used.')
    argpar.add_argument('-m', '--milestone', required=False,
                        help='ID of current milestone.')
    argpar.add_argument('-e', '--extras', dest='extras_filepath', default='',
                        help='File path to sprints')
    args = argpar.parse_args()
    set_logging_level(args.loglevel)

    user, token = fetch_credentials()
    endpoint = config('JIRA_API_ENDPOINT', default=DEFAULT_JIRA_URL)
    JUGGLER = JiraJuggler(endpoint, user, token, args.query, args.kids_query, links=args.links)

    JUGGLER.juggle(
        output=args.output,
        depend_on_preceding=args.depend_on_preceding,
        sprint_field_name=args.sprint_field_name,
        weeklymax=args.weeklymax,
        current_date=args.current_date,
        milestone=args.milestone,
        extras_filepath=args.extras_filepath,
    )
    return 0


def entrypoint():
    """Wrapper function of main"""
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
