import datetime
import enum
import logging
import re

import jira
from dateutil import parser


__all__ = ('Sprint', 'SprintAccessor',)


class SprintState(enum.Enum):
    ACTIVE = "ACTIVE"
    FUTURE = "FUTURE"
    CLOSED = "CLOSED"
    UNDEFINED = "UNDEFINED"


class Sprint:
    priorities = {
        SprintState.ACTIVE: 3,
        SprintState.FUTURE: 2,
        SprintState.CLOSED: 1,
        SprintState.UNDEFINED: 0,
    }

    def __init__(self, name: str, state: SprintState, start_date: datetime.datetime | None):
        self.name = name
        self.state = state
        self.start_date = start_date

    @property
    def priority(self):
        return self.priorities[self.state]


class SprintAccessor:

    def __init__(
            self,
            sprint_field_name: str,
            sprint_re_pattern: str,
            sprint_re_repl: str,
            extras: dict
    ):
        self._sprint_field_name = sprint_field_name
        self._pattern = re.compile(sprint_re_pattern)
        self._sprint_re_repl = sprint_re_repl
        self._extras = extras

    def __call__(self, jira_issue: jira.Issue):
        sprint = Sprint("", SprintState.UNDEFINED, None)
        if jira_issue.key in self._extras:
            sprint_name = self._extras[jira_issue.key].sprint
            if sprint_name is not None:
                return Sprint(sprint_name, SprintState.UNDEFINED, None)
        values = getattr(jira_issue.fields, self._sprint_field_name, None)
        if values is not None:
            if isinstance(values, str):
                values = [values]
            for sprint_info in values:
                if isinstance(sprint_info, (str, bytes)):  # Jira Server
                    state_match = re.search("state=({})".format("|".join([k.value for k in Sprint.priorities])), sprint_info)
                    if state_match:
                        state = SprintState(state_match.group(1))
                        prio = Sprint.priorities[state]
                        if prio > sprint.priority:
                            name = re.search("name=(.+?),", sprint_info).group(1)
                            if self._pattern.fullmatch(name):
                                name = self._pattern.sub(self._sprint_re_repl, name)
                                sprint = Sprint(
                                    name,
                                    state,
                                    self._extract_start_date(sprint_info, jira_issue.key)
                                )

                else:  # Jira Cloud
                    state = SprintState(sprint_info.state.upper())
                    if state in Sprint.priorities:
                        prio = Sprint.priorities[state]
                        if prio > sprint.priority:
                            name = sprint_info.name
                            if self._pattern.fullmatch(name):
                                name = self._pattern.sub(self._sprint_re_repl, name)
                                sprint = Sprint(
                                    name,
                                    state,
                                    parser.parse(sprint_info.startDate) if hasattr(sprint_info, 'startDate') else None
                                )
        return sprint

    @staticmethod
    def _extract_start_date(sprint_info, issue_key):
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
