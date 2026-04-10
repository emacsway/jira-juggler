from mlx.jira_juggler.tasks.base_task import JugglerTask
from mlx.jira_juggler.tasks.properties.registry import Registry


__all__ = ('Epic',)


class Epic(JugglerTask):

    def load_from_jira_issue(self, jira_issue):
        super().load_from_jira_issue(jira_issue)

        if not self.properties['time'].is_empty and self.properties['time'].name == 'fact:end' and not self.is_complete():
            # Actually the PBI is not finished yet.
            self.properties['time'].clear()
        if not self.properties['time'].is_empty and self.properties['time'].name == 'fact:start' and self.todo():
            # Actually the PBI is not started yet.
            self.properties['time'].clear()
        if not self.properties['time'].is_empty and self.properties['time'].name == 'start' and not self.todo() and self.children:
            # Started with the first PBI.
            self.properties['time'].clear()


    def shift_unstarted_tasks_to_milestone(self, milestone):
        for child in self.children:
            child.shift_unstarted_tasks_to_milestone(milestone)

    def collect_todo_tasks(self, collector: Registry):
        for child in self.children:
            child.collect_todo_tasks(collector)
