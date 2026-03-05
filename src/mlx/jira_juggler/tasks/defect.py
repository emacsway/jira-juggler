from mlx.jira_juggler.tasks.pbi import BacklogItem
from mlx.jira_juggler.tasks.properties.effort import PertEstimate

__all__ = (
    'DefectMixin',
    'Defect',
)


class DefectMixin:
    def load_from_jira_issue(self, jira_issue):
        # if jira_issue.pert.expected_duration == 0:
        #     jira_issue.pert = PertEstimate(0.3, 0.5, 2.5)
        super().load_from_jira_issue(jira_issue)
        if self.properties['effort'].is_empty:
            self.properties['effort'].update(PertEstimate(0.2, 0.4, 2))


class Defect(DefectMixin, BacklogItem):
    pass
