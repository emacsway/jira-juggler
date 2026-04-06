import datetime
import logging
from _operator import attrgetter

from dateutil import parser

from mlx.jira_juggler.utils.date import to_juggler_date
from mlx.jira_juggler.tasks.properties.base_property import JugglerTaskProperty
from mlx.jira_juggler.tasks.properties.constants import (
    TODO_STATUSES,
    PROGRESS_STATUSES,
    DEVELOPED_STATUSES,
    RESOLVED_STATUSES,
    DONE_STATUSES,
    PENDING_STATUSES,
)


__all__ = ('JugglerTaskTime',)


class JugglerTaskTime(JugglerTaskProperty):
    """Class for setting the start/end time of a juggler task"""

    DEFAULT_VALUE = ''
    PREFIX = ''

    def load_from_jira_issue(self, jira_issue):
        start = self.do_get_start_date(jira_issue)
        fact_start = self.do_determine_fact_start_date(jira_issue)
        fact_end = self.do_determine_fact_end_date(jira_issue)
        logging.debug("""Date: %s %r %r %r""", jira_issue.key, start, fact_start, fact_end)
        if fact_end is not None and fact_start is not None and fact_start > fact_end:
            # It's a reopened task
            fact_end = None
        if fact_end:
            if jira_issue.fields.status.name.lower() not in (TODO_STATUSES + PROGRESS_STATUSES):
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
