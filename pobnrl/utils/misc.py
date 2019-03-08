""" miscellaneous functions """

import numpy as np


def epsilon_greedy(q_values, epsilon, action_space) -> int:
    """epsilon_greedy returns epsilon greedy action

    :param q_values: a list of q values, one for each action
    :param epsilon: the chances of picking a random action
    :param action_space: a space of actions to sample from
    :rtype: int the action to pick (assumes to be discrete)

    """
    return np.argmax(q_values) if np.random.random(
    ) > epsilon else action_space.sample()


def linear_interpolation(left, right, alpha):
    """ linear interpolation between l and r with alpha """
    return left + alpha * (right - left)


class PiecewiseSchedule(object):
    """ scheduler advancing piecewise """

    def __init__(
            self,
            endpoints,
            interpolation=linear_interpolation,
            outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
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
            if l_t <= time and time < r_t:
                alpha = float(time - l_t) / (r_t - l_t)
                return self._interpolation(left, right, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value
