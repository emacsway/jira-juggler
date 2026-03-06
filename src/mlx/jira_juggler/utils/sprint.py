import logging
import re

from dateutil import parser


__all__ = ('Sprint', 'make_sprint_accessor', 'extract_start_date',)


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
                            if pattern.fullmatch(name):
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
                            if pattern.fullmatch(name):
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
