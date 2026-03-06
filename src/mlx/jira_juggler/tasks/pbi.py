from mlx.jira_juggler.tasks.subtasks.qa_manual_subtask import QaManualSubtask
from mlx.jira_juggler.tasks.subtasks.qa_auto_subtask import QaAutoSubtask
from mlx.jira_juggler.tasks.base_task import JugglerTask
from mlx.jira_juggler.utils.identifier import to_identifier


__all__ = ('BacklogItem',)


class BacklogItem(JugglerTask):

    def load_from_jira_issue(self, jira_issue):
        super().load_from_jira_issue(jira_issue)

        manual_testing_tasks = [child for child in self.children if isinstance(child, QaManualSubtask)]
        if manual_testing_tasks:
            manual_testing_dependencies = [
                child for child in self.children if not isinstance(child, (QaAutoSubtask, QaManualSubtask))
            ]
            for manual_testing_task in manual_testing_tasks:
                if manual_testing_task.time_is_empty():
                    for manual_testing_dependency in manual_testing_dependencies:
                        manual_testing_task.properties['depends'].append_value(
                            to_identifier(manual_testing_dependency.key)
                        )

        if not self.properties['time'].is_empty and self.properties['time'].name == 'fact:end' and not self.is_complete():
            # Actually the PBI is not finished yet.
            self.properties['time'].clear()
        if not self.properties['time'].is_empty and self.properties['time'].name == 'fact:start' and self.todo():
            # Actually the PBI is not started yet.
            self.properties['time'].clear()

    def shift_unstarted_tasks_to_milestone(self, extras, milestone):
        if not self.is_dor():
            milestone = "${sprint_non_dor}"
        super().shift_unstarted_tasks_to_milestone(extras, milestone)

    def adjust_priority(self, extras):
        priority = None
        if self.key in extras:
            priority = extras[self.key].priority
        if priority is not None:
            self.properties['priority'].value = priority
            if self.children:
                for child in self.children:
                    child.adjust_priority(extras)
        elif not self.is_dor() and self.todo():
            self.properties['priority'].value = 1
        elif self.children:
            for child in self.children:
                child.adjust_priority(extras)
