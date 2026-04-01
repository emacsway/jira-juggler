from mlx.jira_juggler.tasks.subtasks.subtask import BaseSubtask
from mlx.jira_juggler.tasks.properties.effort import PertEstimate


__all__ = ('QaAutoSubtask',)


class QaAutoSubtask(BaseSubtask):
    def load_from_jira_issue(self, jira_issue):
        super().load_from_jira_issue(jira_issue)
        if self.properties['effort'].is_empty:
            self.properties['effort'].update(PertEstimate(0.3, 0.7, 1.0))
