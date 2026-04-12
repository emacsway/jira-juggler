import datetime
import enum
import logging
import re

import jira
from dateutil import parser, tz


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

    def __init__(
            self,
            name: str,
            state: SprintState = SprintState.UNDEFINED,
            start: datetime.datetime | None = None,
            end: datetime.datetime | None = None  # FIXME: Remove me?
    ):
        self.name = name
        self.state = state
        self.start = start
        self.end = end

    @property
    def priority(self):
        return self.priorities[self.state]

    def __contains__(self, dt: datetime.datetime) -> bool:
        if self.start is None or self.end is None:
            return False
        return self.start <= dt < self.end


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
        self._sprint_length = datetime.timedelta(weeks=2)

    def __call__(self, jira_issue: jira.Issue):
        sprint = Sprint("")
        if jira_issue.key in self._extras:
            sprint_name = self._extras[jira_issue.key].sprint
            if sprint_name is not None:
                return Sprint(sprint_name)
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
                                start_dt = self._extract_start_date(sprint_info, jira_issue.key)
                                end_dt = self._adjust_end_date(start_dt, self._extract_end_date(
                                    sprint_info, jira_issue.key
                                ))
                                sprint = Sprint(name, state, start_dt, end_dt)

                else:  # Jira Cloud
                    state = SprintState(sprint_info.state.upper())
                    if state in Sprint.priorities:
                        prio = Sprint.priorities[state]
                        if prio > sprint.priority:
                            name = sprint_info.name
                            if self._pattern.fullmatch(name):
                                name = self._pattern.sub(self._sprint_re_repl, name)
                                start_dt = self._parse_datetime(sprint_info.startDate) if hasattr(sprint_info, 'startDate') else None
                                end_dt = self._adjust_end_date(
                                    start_dt,
                                    self._parse_datetime(sprint_info.endDate) if hasattr(sprint_info, 'endDate') else None
                                )
                                sprint = Sprint(name, state, start_dt, end_dt)
        return sprint

    def _extract_start_date(self, sprint_info, issue_key):
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
                    return self._parse_datetime(start_date_match.group(1))
                except parser.ParserError as err:
                    logging.debug("Failed to parse start date of sprint of issue %s: %s", issue_key, err)
                    return None

    def _extract_end_date(self, sprint_info, issue_key):
        """Extracts the end date from the given info string.

        Args:
            sprint_info (str): Raw information about a sprint, as returned by the JIRA API
            issue_key (str): Name of the JIRA issue

        Returns:
            datetime.datetime/None: Start date as a datetime.datetime object or None if the sprint does not have a end date
        """
        end_date_match = re.search("endDate=(.+?),", sprint_info)
        if end_date_match:
            end_date_str = end_date_match.group(1)
            if end_date_str != '<null>':
                try:
                    return self._parse_datetime(end_date_match.group(1))
                except parser.ParserError as err:
                    logging.debug("Failed to parse end date of sprint of issue %s: %s", issue_key, err)
                    return None

    @staticmethod
    def _parse_datetime(dt_str: str):
        dt = parser.parse(dt_str)
        dt = dt.astimezone(tz.tzutc())
        dt = dt.replace(hour=8, minute=0, second=0, microsecond=0)
        return dt

    def _adjust_end_date(self, start_dt: datetime.datetime | None, end_dt: datetime.datetime | None) -> datetime.datetime | None:
        if start_dt is not None:
            if end_dt is None:
                return start_dt + self._sprint_length
            elif end_dt - start_dt < self._sprint_length:
                return start_dt + self._sprint_length
        return end_dt
