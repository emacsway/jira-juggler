from mlx.jira_juggler.tasks.defect import DefectMixin
from mlx.jira_juggler.tasks.subtasks.subtask import BaseSubtask


__all__ = ('DefectSubtask',)


class DefectSubtask(DefectMixin, BaseSubtask):
    pass
