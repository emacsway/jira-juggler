import abc
import functools
import logging
import math
import operator
import typing

from mlx.jira_juggler.tasks.properties.base_property import JugglerTaskProperty
from mlx.jira_juggler.tasks.properties.constants import DONE_STATUSES, RESOLVED_STATUSES, PENDING_STATUSES, TAB

__all__ = (
    'IPertEstimate',
    'EmptyPertEstimate',
    'PertEstimate',
    'CompositePertEstimate',
    'JugglerTaskEffort',
)


class IPertEstimate(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def optimistic(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nominal(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pessimistic(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def expected_duration(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def standard_deviation(self) -> float:
        raise NotImplementedError


class EmptyPertEstimate(IPertEstimate):

    @property
    def optimistic(self) -> float:
        return 0

    @property
    def nominal(self) -> float:
        return 0

    @property
    def pessimistic(self) -> float:
        return 0

    @property
    def expected_duration(self) -> float:
        return 0

    @property
    def standard_deviation(self) -> float:
        return 0


class PertEstimate:
    MINIMAL_VALUE = 1.0 / 8
    _optimistic: float
    _nominal: float
    _pessimistic: float

    def __init__(self, optimistic: float, nominal: float, pessimistic: float):
        assert optimistic is not None
        assert nominal is not None
        assert nominal >= optimistic
        assert pessimistic is not None
        assert pessimistic >= nominal

        self._optimistic = max(optimistic, self.MINIMAL_VALUE)
        self._nominal = max(nominal, self.MINIMAL_VALUE)
        self._pessimistic = max(pessimistic, self.MINIMAL_VALUE)

    @property
    def optimistic(self) -> float:
        return self._optimistic

    @property
    def nominal(self) -> float:
        return self._nominal

    @property
    def pessimistic(self) -> float:
        return self._pessimistic

    @property
    def expected_duration(self) -> float:
        return (self.optimistic + 4*self.nominal + self.pessimistic) / 6

    @property
    def standard_deviation(self) -> float:
        return (self.pessimistic - self.optimistic) / 6


class CompositePertEstimate(IPertEstimate):
    _children: typing.Iterable[IPertEstimate]

    def __init__(self, children: typing.Iterable[IPertEstimate]):
        self._children = children

    @property
    def optimistic(self) -> float:
        return functools.reduce(
            operator.add,
            map(operator.attrgetter('optimistic'), self._children),
            0
        )

    @property
    def nominal(self) -> float:
        return functools.reduce(
            operator.add,
            map(operator.attrgetter('nominal'), self._children),
            0
        )

    @property
    def pessimistic(self) -> float:
        return functools.reduce(
            operator.add,
            map(operator.attrgetter('pessimistic'), self._children),
            0
        )

    @property
    def expected_duration(self) -> float:
        return functools.reduce(
            operator.add,
            map(operator.attrgetter('expected_duration'), self._children),
            0
        )

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(functools.reduce(
            operator.add,
            map(
                functools.partial(pow, exp=2),
                map(operator.attrgetter('expected_duration'), self._children)
            ),
            0
        ))


class JugglerTaskEffort(JugglerTaskProperty):
    """Class for the effort (estimate) of a juggler task"""

    # For converting the seconds (Jira) to days
    UNIT = 'd'
    FACTOR = 8.0 * 60 * 60

    DEFAULT_NAME = 'effort'
    MINIMAL_VALUE = 1.0 / 8
    DEFAULT_VALUE = MINIMAL_VALUE
    SUFFIX = UNIT
    pert: IPertEstimate = EmptyPertEstimate()

    def load_from_jira_issue(self, jira_issue):
        """Loads the object with data from a Jira issue

        Args:
            jira_issue (jira.resources.Issue): The Jira issue to load from
        """
        self.pert = jira_issue.pert
        if self.pert.expected_duration:
            self.value = round(self.pert.expected_duration, 3)
        elif hasattr(jira_issue.fields, 'timeoriginalestimate'):
            estimated_time = jira_issue.fields.timeoriginalestimate
            if estimated_time is not None:
                self.value = estimated_time / self.FACTOR
                logged_time = jira_issue.fields.timespent if jira_issue.fields.timespent else 0
                if jira_issue.fields.status.name.lower() in DONE_STATUSES + RESOLVED_STATUSES + PENDING_STATUSES:
                    # resolved ticket: prioritize Logged time over Estimated
                    if logged_time:
                        self.value = logged_time / self.FACTOR
                elif jira_issue.fields.timeestimate is not None:
                    # open ticket prioritize Remaining time over Estimated
                    if jira_issue.fields.timeestimate:
                        self.value = jira_issue.fields.timeestimate / self.FACTOR
                    else:
                        self.value = self.MINIMAL_VALUE
            else:
                self.value = self.DEFAULT_VALUE
        else:
            self.value = self.DEFAULT_VALUE
            logging.warning('No estimate found for %s, assuming %s%s', jira_issue.key, self.DEFAULT_VALUE, self.UNIT)

    def update(self, pert: IPertEstimate):
        self.pert = pert
        self.value = round(self.pert.expected_duration, 3)

    def validate(self, task, tasks):
        """Validates (and corrects) the current task property

        Args:
            task (JugglerTask): Task to which the property belongs
            tasks (list): Modifiable list of JugglerTask instances to which the current task belongs. Will be used to
                verify relations to other tasks.
        """
        if self.value == 0:
            logging.warning('Estimate for %s, is 0. Excluding', task.key)
            tasks.remove(task)
        elif self.value < self.MINIMAL_VALUE:
            logging.warning('Estimate %s%s too low for %s, assuming %s%s', self.value, self.UNIT, task.key, self.MINIMAL_VALUE, self.UNIT)
            self.value = self.MINIMAL_VALUE

    def __str__(self):
        result = super().__str__()
        result += self.TEMPLATE.format(
            prop='stdev',
            value=self.VALUE_TEMPLATE.format(
                prefix='',
                value=round(self.pert.standard_deviation, 3),
                suffix='d'
            )
        )
        if not isinstance(self.pert, CompositePertEstimate):
            result += TAB + """${pert "%s" "%s" "%s"}\n""" % (
                self.pert.optimistic or self.MINIMAL_VALUE,
                self.pert.nominal or self.MINIMAL_VALUE,
                self.pert.pessimistic or self.MINIMAL_VALUE
            )
        return result
