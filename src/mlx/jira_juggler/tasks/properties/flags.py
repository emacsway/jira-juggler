import re

from mlx.jira_juggler.tasks.properties.base_property import JugglerTaskProperty


__all__ = ('JugglerTaskFlags',)


class JugglerTaskFlags(JugglerTaskProperty):
    """Class for linking of a juggler task"""

    DEFAULT_NAME = 'flags'
    DEFAULT_VALUE = []

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
        if hasattr(jira_issue.fields, 'labels'):
            for label in jira_issue.fields.labels:
                if not self._add_label(label):
                    continue
                identifier = re.sub(r'[^a-zA-Z0-9_]', '_', label)
                if identifier and identifier[0].isdigit():
                    identifier = '_' + identifier
                if identifier:
                    self.append_value(identifier)

    def _add_label(self, label: str) -> bool:
        if label == 'TaskJuggler':
            return False
        if label.startswith('Team'):
            return False
        return True

    def validate(self, task, tasks):
        pass

    def __str__(self):
        """Converts task property object to the task juggler syntax

        Returns:
            str: String representation of the task property in juggler syntax
        """
        if self.value:
            valstr = ''
            for val in self.value:
                if valstr:
                    valstr += ', '
                valstr += self.VALUE_TEMPLATE.format(prefix=self.PREFIX,
                                                     value=val,
                                                     suffix=self.SUFFIX)
            return self.TEMPLATE.format(prop=self.name,
                                        value=valstr)
        return ''
