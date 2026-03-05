import jira

from mlx.jira_juggler.tasks.properties.base_property import JugglerTaskProperty
from mlx.jira_juggler.tasks.properties.registry import Registry
from mlx.jira_juggler.utils.identifier import to_identifier


__all__ = ('JugglerTaskDepends',)


class JugglerTaskDepends(JugglerTaskProperty):
    """Class for linking of a juggler task"""

    DEFAULT_NAME = 'depends'
    DEFAULT_VALUE = []
    PREFIX = ''
    links = set()
    registry: Registry

    def __init__(self, registry: Registry, jira_issue: jira.Issue | None = None):
        self.registry = registry
        super().__init__(jira_issue)

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
        values = [self.registry.path(val) for val in self.value if val in self.registry or self._is_milestone(val)]
        if values:
            valstr = ''
            for val in self.value:
                if val in self.registry:
                    val = self.registry.path(val)
                elif not self._is_milestone(val):
                    continue
                if valstr:
                    valstr += ', '
                valstr += self.VALUE_TEMPLATE.format(prefix=self.PREFIX,
                                                     value=val,
                                                     suffix=self.SUFFIX)
            return self.TEMPLATE.format(prop=self.name,
                                        value=valstr)
        return ''

    @staticmethod
    def _is_milestone(key: str):
        return key.startswith('deliveries.') or key.startswith('${')
