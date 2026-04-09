import datetime
import logging
import re
from _operator import attrgetter

from dateutil import parser
from mlx.jira_juggler.tasks.properties.allocate import JugglerTaskAllocate
from mlx.jira_juggler.tasks.properties.complete import JugglerTaskComplete
from mlx.jira_juggler.tasks.properties.constants import TAB, DONE_STATUSES, RESOLVED_STATUSES, PENDING_STATUSES
from mlx.jira_juggler.tasks.properties.depends import JugglerTaskDepends
from mlx.jira_juggler.tasks.properties.effort import CompositePertEstimate, JugglerTaskEffort
from mlx.jira_juggler.tasks.properties.fact_depends import JugglerTaskFactDepends
from mlx.jira_juggler.tasks.properties.flags import JugglerTaskFlags
from mlx.jira_juggler.tasks.properties.priority import JugglerTaskPriority
from mlx.jira_juggler.tasks.properties.registry import Registry
from mlx.jira_juggler.tasks.properties.time import JugglerTaskTime
from mlx.jira_juggler.utils.add_working_days import AddWorkingDays
from mlx.jira_juggler.utils.identifier import to_identifier
from mlx.jira_juggler.utils.sprint import Sprint


__all__ = ('JugglerTask',)


class JugglerTask:
    """Class for a task for Task-Juggler"""

    class TYPE:
        INITIATIVE = 'Initiative'
        EPIC = 'Epic'
        SPIKE = 'Spike'
        STORY = 'Story'
        DEFECT = 'Bug'
        SUBTASK = 'Sub-task'
        IMPROVEMENT = 'Improvement'
        FEATURE = 'New Feature'

    add_working_days: AddWorkingDays
    sprint: Sprint
    focus_factor = 0.7
    children: list['JugglerTask']
    DEFAULT_KEY = 'NOT_INITIALIZED'
    MAX_SUMMARY_LENGTH = 70
    DEFAULT_SUMMARY = 'Task is not initialized'
    TEMPLATE = '''
task {id} "{description}" {{
{tab}Key "{key}"
{props}
{children}
}}
'''

    @classmethod
    def factory(cls, registry, to_username, jira_issue, parent=None) -> 'JugglerTask':
        if jira_issue.fields.issuetype.name == cls.TYPE.INITIATIVE:
            return Initiative(registry, to_username, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.EPIC:
            return Epic(registry, to_username, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.SPIKE or '[spike]' in jira_issue.fields.summary.lower():
            return Spike(registry, to_username, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.SUBTASK and "[qa auto]" in jira_issue.fields.summary.lower():
            return QaAutoSubtask(registry, to_username, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.SUBTASK and "[qa manual]" in jira_issue.fields.summary.lower():
            return QaManualSubtask(registry, to_username, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.SUBTASK and re.match(r"\[[^\[\]]*bug\].+", jira_issue.fields.summary.lower()) != None:
            return DefectSubtask(registry, to_username, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.DEFECT:
            return Defect(registry, to_username, jira_issue, parent)
        elif jira_issue.fields.issuetype.name == cls.TYPE.SUBTASK:
            return Subtask(registry, to_username, jira_issue, parent)
        else:
            return BacklogItem(registry, to_username, jira_issue, parent)

    def __init__(self, registry, to_username, jira_issue=None, parent=None):
        logging.info('Create JugglerTask for %s', jira_issue.key)

        self.key = self.DEFAULT_KEY
        self.summary = self.DEFAULT_SUMMARY
        self.properties = {}
        self.issue = None
        self.type = None
        self._resolved_at_date = None
        self.parent = parent
        self.registry = registry
        self.to_username = to_username

        if jira_issue:
            self.load_from_jira_issue(jira_issue)

    def _inherit_priority(self):
        if self.parent.properties['priority'].value > self.properties['priority'].value:
            self.properties['priority'].value = self.parent.properties['priority'].value

    def load_from_jira_issue(self, jira_issue):
        """Loads the object with data from a Jira issue

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
        """
        self.key = jira_issue.key
        self.issue = jira_issue
        self.type = jira_issue.fields.issuetype.name
        summary = jira_issue.fields.summary.replace('\"', '\\\"')
        self.summary = (summary[:self.MAX_SUMMARY_LENGTH] + '...') if len(summary) > self.MAX_SUMMARY_LENGTH else summary
        if self.is_resolved:
            self.resolved_at_date = self.determine_resolved_at_date()
        self.properties['allocate'] = JugglerTaskAllocate(self.to_username, jira_issue)
        self.properties['effort'] = JugglerTaskEffort(jira_issue)
        self.properties['depends'] = JugglerTaskDepends(self.registry, jira_issue)
        self.properties['fact:depends'] = JugglerTaskFactDepends(self.registry)
        self.properties['time'] = JugglerTaskTime(jira_issue)
        self.properties['complete'] = JugglerTaskComplete(jira_issue)
        self.properties['priority'] = JugglerTaskPriority(jira_issue)
        self.properties['flags'] = JugglerTaskFlags(jira_issue)
        self.children = [JugglerTask.factory(self.registry, self.to_username, child, self) for child in jira_issue.children]
        if len(self.children) > 0:
            self.properties['effort'].update(CompositePertEstimate([child.properties['effort'].pert for child in self.children]))
        self.registry[to_identifier(self.key)] = self
        if hasattr(self, 'sprint_accessor'):
            self.sprint = self.sprint_accessor(jira_issue)

    def validate(self, tasks, property_identifier):
        """Validates (and corrects) the current task

        Args:
            tasks (list): List of JugglerTask instances to which the current task belongs. Will be used to
                verify relations to other tasks.
            property_identifier (str): Identifier of property type
        """
        if self.key == self.DEFAULT_KEY:
            logging.error('Found a task which is not initialized')
        self.properties[property_identifier].validate(self, tasks)

        for task in tasks:
            if task.children:
                self.validate(task.children, property_identifier)

    def __str__(self):
        """Converts the JugglerTask to the task juggler syntax

        Returns:
            str: String representation of the task in juggler syntax
        """
        props = list()
        for k, v in self.properties.items():
            if k in ('effort', 'allocate', 'complete',) and self.children:
                continue
            props.append(str(v))

        props = "".join(props)
        result = self.TEMPLATE.format(
            id=to_identifier(self.key),
            key=self.key,
            tab=TAB,
            description=self.summary.replace('\"', '\\\"'),
            props=props,
            children="".join(map(self._make_indentation, self.children))
        )
        return result

    @staticmethod
    def _make_indentation(task):
        result = []
        for line in str(task).split("\n"):
            if len(line):
                line = TAB + line
            result.append(line)
        return "\n".join(result)

    @property
    def is_resolved(self):
        """bool: True if JIRA issue has been approved/resolved/closed; False otherwise"""
        return self.issue is not None and self.issue.fields.status.name.lower() in (
                DONE_STATUSES + RESOLVED_STATUSES + PENDING_STATUSES
        )

    @property
    def resolved_at_date(self):
        """datetime.datetime: Date and time corresponding to the last transition to the Approved/Resolved status; the
            transition to the Closed status is used as fallback; None when not resolved
        """
        return self._resolved_at_date

    @resolved_at_date.setter
    def resolved_at_date(self, value):
        self._resolved_at_date = value

    def determine_resolved_at_date(self):
        closed_at_date = None
        for change in sorted(self.issue.changelog.histories, key=attrgetter('created'), reverse=True):
            for item in change.items:
                if item.field.lower() == 'status':
                    status = item.toString.lower()
                    if status in RESOLVED_STATUSES:
                        return parser.isoparse(change.created)
                    elif status in DONE_STATUSES + PENDING_STATUSES and closed_at_date is None:
                        closed_at_date = parser.isoparse(change.created)
        return closed_at_date

    def shift_unstarted_tasks_to_milestone(self, extras, milestone):
        sprint = self.get_sprint(extras)
        if sprint:
            if self.time_is_empty():
                self.properties['fact:depends'].append_value(sprint)
            elif self.children:
                for child in self.children:
                    child.shift_unstarted_tasks_to_milestone(extras, sprint)
            """
            self.properties['fact:depends'].append_value(sprint)
            if not self.time_is_empty() and self.in_progress():
                for child in self.children:
                    if child.time_is_empty():
                        child.shift_unstarted_tasks_to_milestone(extras, milestone)
                        # child.shift_unstarted_tasks_to_milestone(extras, sprint)
            """
        elif self.time_is_empty():
            self.properties['fact:depends'].append_value(milestone)
        elif self.children:
            for child in self.children:
                child.shift_unstarted_tasks_to_milestone(extras, milestone)

    def get_sprint(self, extras):
        sprint = None
        if self.key in extras:
            sprint = extras[self.key].sprint
        if not sprint and getattr(self, 'sprint', None):
            sprint = getattr(self, 'sprint').name
        return sprint

    def adjust_priority(self, extras):
        priority = None
        if self.key in extras:
            priority = extras[self.key].priority
        if priority is not None:
            self.properties['priority'].value = priority
        if self.children:
            for child in self.children:
                child.adjust_priority(extras)

    def is_dor(self):
        if self.children:
            return any(child.is_dor() for child in self.children)  # all()?
        else:
            return not self.properties['effort'].is_empty

    def todo(self):
        if self.children:
            return all(child.todo() for child in self.children)
        else:
            return self.properties['complete'].is_empty

    def in_progress(self):
        if self.children:
            if any(child.in_progress() for child in self.children):
                return True
            complete = any(child.is_complete() for child in self.children)
            todo = any(child.todo() for child in self.children)
            return complete and todo
        else:
            return 0 < self.properties['complete'].value < 100

    def is_complete(self):
        if self.children:
            return all(child.is_complete() for child in self.children)
        else:
            return self.properties['complete'].value == 100

    def collect_todo_tasks(self, collector: Registry):
        if self.time_is_empty():
            collector[to_identifier(self.key)] = self
        elif self.children:
            for child in self.children:
                child.collect_todo_tasks(collector)

    def bottom_up_deps(self, visited: set | None = None) -> set['JugglerTask']:
        if visited is None:
            visited = set()
        if self.key in visited:
            return set()
            # raise RuntimeError('Circular dependency has been detected: %r -> %s' % (visited, self.key))
        visited.add(self.key)
        deps = set()  # TODO: Use OrderedSet?
        for dep in self.properties['depends'].value:
            if dep in self.properties['depends'].registry:
                deps.add(self.properties['depends'].registry.get(dep))
        if self.parent:
            deps = deps.union(self.parent.bottom_up_deps(visited))
        for dep in deps:
            deps = deps.union(dep.bottom_up_deps(visited))
        return deps

    def time_is_empty(self) -> bool:
        if not self.properties['time'].is_empty:
            return False
        for child in self.children:
            if not child.time_is_empty():
                return False
        return True

    def max_time(self) -> datetime.datetime | None:
        dts = []
        if not self.properties['time'].is_empty:
            if 'end' in self.properties['time'].name:
                end_time = self.properties['time'].value
            else:
                start_time = self.properties['time'].value
                working_days = self.properties['effort'].value / self.focus_factor
                end_time = self.add_working_days(start_time, working_days)
            dts.append(end_time)
        for child in self.children:
            dt = child.max_time()
            if dt:
                dts.append(dt)
        if dts:
            return max(dts)
        return None

    def fix_time(self):
        if self.children:
            for child in self.children:
                child.fix_time()
        for dep in self.bottom_up_deps():
            if isinstance(dep, Spike):
                if dep.is_resolved:
                    dt = dep.max_time()
                    if dt is not None:
                        if not self.properties['time'].is_empty:
                            if 'start' in self.properties['time'].name:
                                start_time = self.properties['time'].value
                            else:
                                end_time = self.properties['time'].value
                                days_spent = self.properties['effort'].value / self.focus_factor
                                start_time = self.add_working_days(end_time, -days_spent)
                            if dt > start_time:
                                logging.warning(
                                    """Fix time for %s "%s" %s because of resolved dependency %s "%s" %s""",
                                    self.key, self.summary, self.properties['time'],
                                    dep.key, dep.summary, dt
                                )
                                self.properties['time'] = JugglerTaskTime()
                elif dep.time_is_empty():
                    if not self.properties['time'].is_empty:
                        logging.warning(
                            """Fix time for %s "%s" %s because of unresolved dependency %s "%s" with no time""",
                            self.key, self.summary, self.properties['time'],
                            dep.key, dep.summary
                        )
                        self.properties['time'] = JugglerTaskTime()

    def shift_in_progress_to(self, current_date):
        if self.children:
            for child in self.children:
                child.shift_in_progress_to(current_date)
        elif not self.time_is_empty():
            progress = self.properties['complete'].value
            if 0 < progress < 100:
                self.properties['time'].name = 'fact:start'
                days_spent = self.properties['effort'].value / self.focus_factor
                self.properties['time'].value = self.add_working_days(current_date, -days_spent)

    def sort(self, *, key=None, reverse=False):
        if self.children:
            self.children.sort(key=key, reverse=reverse)
            for child in self.children:
                child.sort(key=key, reverse=reverse)


from mlx.jira_juggler.tasks.subtasks.qa_manual_subtask import QaManualSubtask
from mlx.jira_juggler.tasks.subtasks.qa_auto_subtask import QaAutoSubtask
from mlx.jira_juggler.tasks.subtasks.defect_subtask import DefectSubtask
from mlx.jira_juggler.tasks.subtasks.subtask import Subtask
from mlx.jira_juggler.tasks.spike import Spike
from mlx.jira_juggler.tasks.defect import Defect
from mlx.jira_juggler.tasks.pbi import BacklogItem
from mlx.jira_juggler.tasks.epic import Epic
from mlx.jira_juggler.tasks.initiative import Initiative
