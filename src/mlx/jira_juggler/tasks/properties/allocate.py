from _operator import attrgetter

import jira

from mlx.jira_juggler.utils.user import to_username
from mlx.jira_juggler.tasks.properties.base_property import JugglerTaskProperty
from mlx.jira_juggler.tasks.properties.constants import DONE_STATUSES, PENDING_STATUSES, RESOLVED_STATUSES, TAB

__all__ = ('JugglerTaskAllocate',)


class JugglerTaskAllocate(JugglerTaskProperty):
    """Class for the allocation (assignee) of a juggler task"""

    DEFAULT_NAME = 'allocate'
    DEFAULT_VALUE = '"not assigned"'

    def load_from_jira_issue(self, jira_issue: jira.Issue):
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
