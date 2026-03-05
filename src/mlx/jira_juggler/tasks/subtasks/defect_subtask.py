from mlx.jira_juggler.tasks.defect import DefectMixin
from mlx.jira_juggler.tasks.subtasks.subtask import Subtask


__all__ = ('DefectSubtask',)


class DefectSubtask(DefectMixin, Subtask):
    pass
