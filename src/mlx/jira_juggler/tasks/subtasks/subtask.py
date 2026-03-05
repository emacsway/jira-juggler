from mlx.jira_juggler.tasks.base_task import JugglerTask

__all__ = ('Subtask',)


class Subtask(JugglerTask):
    def load_from_jira_issue(self, jira_issue):
        super().load_from_jira_issue(jira_issue)
        # self._inherit_priority()
        # self.properties['priority'].clear()
        self.properties['priority'].set_relatively_on(self.parent.properties['priority'].value)

    def adjust_priority(self, extras):
        super().adjust_priority(extras)
        self.properties['priority'].set_relatively_on(self.parent.properties['priority'].value)
