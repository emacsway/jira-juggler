import jira

__all__ = ('jira_instance',)


# FIXME:
jira_instance: jira.JIRA | None = None
