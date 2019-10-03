""" miscelaneous functionality for agents

* exploration schedules

"""

from typing import List, Callable, Tuple
import abc
import numpy as np

from po_nrl.environments import ActionSpace


def epsilon_greedy(
        q_values: np.array,
        epsilon: float,
        action_space: ActionSpace) -> int:
    """ returns epsilon greedy action

    Args:
         q_values: (`np.array'): a list of q values, one for each action
         epsilon: (`float`): the probability of picking a random action
         action_space: (`pobnrl.environments.ActionSpace`): : a space of actions to sample from

    RETURNS (`int`): an action (assuming discrete domains

    """
    assert 1 >= epsilon >= 0

    if np.random.random() > epsilon:
        return np.argmax(q_values)

    return action_space.sample()


class ExplorationSchedule(abc.ABC):
    """ interface for e-greedy exploration schedule """

    @abc.abstractmethod
    def value(self, time: int) -> float:
        """ returns epsilon for a given timestep """


class FixedExploration(ExplorationSchedule):
    """ always returns 0 probability for exploring """

    def __init__(self, epsilon: float):
        """ creates fixed exploration policy with epsilon

        Args:
             epsilon: (`float`): the e-greedy epsilon exploration chance

        """

        self._epsilon = epsilon

    def value(self, _time: int) -> float:
        """ returns epsilon for a given timestep """
        return self._epsilon


def linear_interpolation(left: float, right: float, alpha: float) -> float:
    """ linear interpolation between l and r with alpha

    Args:
         left: (`float`):
         right: (`float`):
         alpha: (`float`):

    RETURNS (`float`):

    """
    return left + alpha * (right - left)


class PiecewiseSchedule(ExplorationSchedule):
    """ scheduler advancing piecewise """

    def __init__(
            self,
            endpoints: List[Tuple[float, float]],
            interpolation: Callable[[float, float, float], float]
            = linear_interpolation,
            outside_value: float = float('NaN')):
        """ Piecewise Schedule

        Args:
             endpoints: (`List[Tuple[float, float]]`):

                list of pairs `(time, value)` meanining that schedule should
                output `value` when `t==time`. All the values for time must be
                sorted in an increasing order. When t is between two times,
                e.g. `(time_a, value_a)` and `(time_b, value_b)`, such that
                `time_a <= t < time_b` then value outputs
                `interpolation(value_a, value_b, alpha)` where alpha is a
                fraction of time passed between `time_a` and `time_b` for time
                `t`.

             interpolation (`Callable[[float, float, float], float]`):

                a function that takes value to the left and to the right of t
                according to the `endpoints`. Alpha is the fraction of distance
                from left endpoint to right endpoint that t has covered. See
                linear_interpolation for example.

             outside_value (`float`):

                if the value is requested outside of all the intervals
                sepecified in `endpoints` this value is returned. If None then
                AssertionError is raised when outside value is requested.

        """

        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, time):
        """See Schedule.value"""
        for (l_t, left), (r_t, right) in zip(
                self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= time < r_t:
                alpha = float(time - l_t) / (r_t - l_t)
                return self._interpolation(left, right, alpha)

        # t does not belong to any of the pieces, so doom.
        assert not np.isnan(self._outside_value)
        return self._outside_value
