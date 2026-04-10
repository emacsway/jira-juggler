#! /usr/bin/python3
"""
Jira to task-juggler extraction script

This script queries Jira, and generates a task-juggler input file to generate a Gantt chart.
"""
import argparse
import csv
import logging
import re
import datetime
from functools import cmp_to_key
from itertools import chain
from pathlib import Path

import jira
from dateutil import parser, tz
from decouple import config
from natsort import natsorted, ns

from mlx.jira_juggler.tasks.base_task import JugglerTask
from mlx.jira_juggler.tasks.properties.depends import JugglerTaskDepends
from mlx.jira_juggler.tasks.properties.effort import (
    DailyMax,
    WeeklyMax,
    EmptyPertEstimate,
    PertEstimate,
    CompositePertEstimate,
    JugglerTaskEffort,
)
from mlx.jira_juggler.tasks.properties.registry import Registry
from mlx.jira_juggler.utils.add_working_days import AddWorkingDays
from mlx.jira_juggler.utils.auth import fetch_credentials
from mlx.jira_juggler.utils.identifier import to_identifier
from mlx.jira_juggler.utils.sprint import SprintAccessor
from mlx.jira_juggler.utils.user import ToUsername

DEFAULT_LOGLEVEL = 'warning'
DEFAULT_JIRA_URL = 'https://melexis.atlassian.net'
DEFAULT_OUTPUT = 'jira_export.tji'

JIRA_PAGE_SIZE = 50


def set_logging_level(loglevel):
    """Sets the logging level

    Args:
        loglevel (str): String representation of the loglevel
    """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)


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


class JiraJuggler:
    """Class for task-juggling Jira results"""
    _jira_instance: jira.JIRA
    _to_username: ToUsername
    _registry: Registry

    def __init__(self, endpoint, user, token, query, kids_query,  links=None):
        """Constructs a JIRA juggler object

        Args:
            endpoint (str): Endpoint for the Jira Cloud (or Server)
            user (str): Email address (or username)
            token (str): API token (or password)
            query (str): The query to run
            links (set/None): List of issue link type inward/outward links; None to use the default configuration
        """
        logging.info('Jira endpoint: %s', endpoint)

        self._jira_instance = jira.JIRA(endpoint, basic_auth=(user, token), options={'rest_api_version': 3})
        self._to_username = ToUsername(self._jira_instance)
        self._registry = Registry()
        if 'ORDER BY' not in query.upper():
            query = "%s ORDER BY priority DESC, created ASC" % query
        logging.info('Query: %s', query)
        self.query = query
        self.kids_query = kids_query

        all_jira_link_types = self._jira_instance.issue_link_types()
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
        tasks = [JugglerTask.factory(self._registry, self._to_username, issue) for issue in self._load_issues(self.query)]
        self.validate_tasks(tasks)
        if sprint_field_name:
            self.sort_tasks_on_sprint(tasks)
        tasks.sort(key=cmp_to_key(self.compare_status))
        if depend_on_preceding:
            self.link_to_preceding_task(tasks, **kwargs)
        return tasks

    def _load_issues(self, query):
        next_page_token = None
        result: list[jira.Issue] = []
        while True:
            try:
                response = self._jira_instance.enhanced_search_issues(
                    query,
                    maxResults=JIRA_PAGE_SIZE,
                    expand='changelog',
                    nextPageToken=next_page_token
                )
            except jira.JIRAError as err:
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
            if issue.key in self._extras:
                flags = self._extras[issue.key].flags
                for flag in flags:
                    issue.fields.labels.append(flag)
                priority = self._extras[issue.key].priority
                if priority is not None:
                    issue.fields.priority.tj_value = priority

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
            pert_response = self._jira_instance.issue_property(issue.key, 'pert-estimation')
        except jira.JIRAError as e:
            if e.status_code == 404:
                return EmptyPertEstimate()
            else:
                raise
        else:
            limits = []
            if getattr(pert_response.value, 'dailymax', None):
                limits.append(DailyMax(pert_response.value.dailymax))
            if getattr(pert_response.value, 'weeklymax', None):
                limits.append(WeeklyMax(pert_response.value.weeklymax))
            return PertEstimate(
                optimistic=pert_response.value.optimistic,
                nominal=pert_response.value.nominal,
                pessimistic=pert_response.value.pessimistic,
                limits=limits,
            )

    def juggle(self, output=None, **kwargs):
        """Queries JIRA and generates task-juggler output from given issues

        Args:
            list: A list of JugglerTask instances
        """
        self._extras = {}
        if kwargs.get('extras_filepath'):
            self._extras = self._load_extras(kwargs['extras_filepath'])
        extras = self._extras

        JugglerTask.add_working_days = staticmethod(AddWorkingDays(kwargs.get('weeklymax')))
        if kwargs.get('sprint_field_name'):
            kwargs['sprint_field_name'], sprint_re_pattern, sprint_re_repl = kwargs['sprint_field_name']
            JugglerTask.sprint_accessor = staticmethod(SprintAccessor(
                kwargs['sprint_field_name'],
                sprint_re_pattern,
                sprint_re_repl,
                self._extras
            ))

        juggler_tasks = self.load_issues_from_jira(**kwargs)
        if not juggler_tasks:
            return None

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
                task.shift_unstarted_tasks_to_milestone(kwargs['milestone'])
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

    def sort_tasks_on_sprint(self, tasks):
        """Sorts given list of tasks based on the values of the field with the given name.

        JIRA issues that are not assigned to a sprint will be ordered last.

        Args:
            tasks (list): List of JugglerTask instances to sort in place
        """
        for task in tasks:
            if task.children:
                self.sort_tasks_on_sprint(task.children)

        logging.debug("Sorting tasks based on sprint information...")
        tasks.sort(key=cmp_to_key(self.compare_sprint_priority))

    @staticmethod
    def compare_sprint_priority(a: JugglerTask, b: JugglerTask):
        """Compares the priority of two tasks based on the sprint information

        The sprint_priority attribute is taken into account first, followed by the sprint_start_date and, lastly, the
        sprint_name attribute using natural sorting (a sprint with the word 'backlog' in its name is sorted as last).

        Args:
            a (JugglerTask): First JugglerTask instance in the comparison
            b (JugglerTask): Second JugglerTask instance in the comparison

        Returns:
            int: 0 for equal priority; -1 to prioritize a over b; 1 otherwise
        """
        if a.sprint.priority > b.sprint.priority:
            return -1
        if a.sprint.priority < b.sprint.priority:
            return 1
        if a.sprint.priority == 0 or a.sprint.name == b.sprint.name:
            return 0  # no/same sprint associated with both issues
        if type(a.sprint.start_date) != type(b.sprint.start_date):  # noqa
            return -1 if b.sprint.start_date is None else 1
        if a.sprint.start_date == b.sprint.start_date:
            # a sprint with backlog in its name has lower priority
            if "backlog" not in a.sprint.name.lower() and "backlog" in b.sprint.name.lower():
                return -1
            if "backlog" in a.sprint.name.lower() and "backlog" not in b.sprint.name.lower():
                return 1
            if natsorted([a.sprint.name, b.sprint.name], alg=ns.IGNORECASE)[0] == a.sprint.name:
                return -1
            return 1
        if a.sprint.start_date < b.sprint.start_date:
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
    argpar.add_argument('-s', '--sort-on-sprint', dest='sprint_field_name', default='', nargs=3,
                        help='Sort unresolved tasks by using field name that stores sprint(s), e.g. customfield_10851, '
                             'in addition to the original order')
    argpar.add_argument('-w', '--weeklymax', default=5, type=int,
                        help='Number of allocated workdays per week used to approximate '
                             'start time of unresolved tasks with logged time')
    argpar.add_argument('-c', '--current-date', default=datetime.datetime.now(tz.tzutc()).replace(minute=0, second=0, microsecond=0), type=parser.isoparse,
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
