from mlx.jira_juggler.tasks.base_task import JugglerTask
from mlx.jira_juggler.tasks.properties.registry import Registry


__all__ = ('Epic',)


class Epic(JugglerTask):

    def shift_unstarted_tasks_to_milestone(self, extras, milestone):
        for child in self.children:
            child.shift_unstarted_tasks_to_milestone(extras, milestone)

    def collect_todo_tasks(self, collector: Registry):
        for child in self.children:
            child.collect_todo_tasks(collector)
