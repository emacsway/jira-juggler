from mlx.jira_juggler.tasks.base_task import JugglerTask
from mlx.jira_juggler.tasks.properties.effort import PertEstimate


__all__ = ('Spike',)


class Spike(JugglerTask):
    def load_from_jira_issue(self, jira_issue):
        super().load_from_jira_issue(jira_issue)
        if self.properties['effort'].is_empty:
            self.properties['effort'].update(PertEstimate(0.2, 0.4, 0.7))
