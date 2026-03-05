import math
from _operator import attrgetter

from mlx.jira_juggler.tasks.properties.base_property import JugglerTaskProperty
from mlx.jira_juggler.tasks.properties.constants import (
    DEVELOPED_STATUSES,
    DONE_STATUSES,
    RESOLVED_STATUSES,
    PENDING_STATUSES,
    PROGRESS_STATUSES,
)


__all__ = ('JugglerTaskComplete',)


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
