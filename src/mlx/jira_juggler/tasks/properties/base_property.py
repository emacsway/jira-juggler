import copy
from abc import ABC

__all__ = ('JugglerTaskProperty', 'TAB',)

TAB = ' ' * 2


class JugglerTaskProperty(ABC):
    """Class for a property of a Task Juggler"""

    DEFAULT_NAME = 'property name'
    DEFAULT_VALUE = 'not initialized'
    PREFIX = ''
    SUFFIX = ''
    TEMPLATE = TAB + '{prop} {value}\n'
    VALUE_TEMPLATE = '{prefix}{value}{suffix}'

    def __init__(self, jira_issue=None):
        """Initializes the task juggler property

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
            value (object): Value of the property
        """
        self.name = self.DEFAULT_NAME
        self.value = copy.copy(self.DEFAULT_VALUE)

        if jira_issue:
            self.load_from_jira_issue(jira_issue)

    @property
    def is_empty(self):
        """bool: True if the property contains an empty or uninitialized value"""
        return not self.value or self.value == self.DEFAULT_VALUE

    def clear(self):
        """Sets the name and value to the default"""
        self.name = self.DEFAULT_NAME
        self.value = self.DEFAULT_VALUE

    def load_from_jira_issue(self, jira_issue):
        """Loads the object with data from a Jira issue

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
        """

    def validate(self, task, tasks):
        """Validates (and corrects) the current task property

        Args:
            task (JugglerTask): Task to which the property belongs
            tasks (list): List of JugglerTask instances to which the current task belongs. Will be used to
                verify relations to other tasks.
        """

    def __str__(self):
        """Converts task property object to the task juggler syntax

        Returns:
            str: String representation of the task property in juggler syntax
        """
        if self.value is not None:
            return self.TEMPLATE.format(prop=self.name,
                                        value=self.VALUE_TEMPLATE.format(prefix=self.PREFIX,
                                                                         value=self.value,
                                                                         suffix=self.SUFFIX))
        return ''
