import jira
from mlx.jira_juggler.tasks.properties.base_property import JugglerTaskProperty

__all__ = ('JugglerTaskPriority',)


class JugglerTaskPriority(JugglerTaskProperty):
    """Class for the allocation (assignee) of a juggler task"""

    _PRIORITY_MAPPING = {
        'lowest': 300,
        'low': 400,
        'medium': 500,
        'high': 600,
        'highest': 700,
        'critical': 800,
    }

    DEFAULT_NAME = 'priority'
    DEFAULT_VALUE = _PRIORITY_MAPPING['medium']

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self._leaf_value = value

    def set_relatively_on(self, parent_priority=None):
        if parent_priority is not None:
            relative_priority = round((self._leaf_value - self._PRIORITY_MAPPING['medium']) * 0.03)
            if relative_priority != 0:
                self._value = parent_priority + relative_priority

    def load_from_jira_issue(self, jira_issue: jira.Issue):
        if jira_issue.fields.priority:
            tj_value = getattr(jira_issue.fields.priority, 'tj_value', None)
            if tj_value is not None:
                self.value = tj_value
            else:
                self.value = self._PRIORITY_MAPPING[jira_issue.fields.priority.name.lower()]

    def __str__(self):
        if self.value != self.DEFAULT_VALUE:
            return super().__str__()
        return ''
