""" miscellaneous functions

sample functionality

mathematical spaces or sets

wrapper functions for interaction with tensorflow:

    Simply has a 'init' and 'getter' function to initiate and return sessions
"""

# TODO fix documentation here in header

from typing import List

import numpy as np
import tensorflow as tf


def epsilon_greedy(q_values, epsilon: float, action_space) -> int:
    """ returns epsilon greedy action

    Args:
         q_values: a list of q values, one for each action
         epsilon: (`float`): the probability of picking a random action
         action_space: a space of actions to sample from

    RETURNS (`int`): an action (assuming discrete environments)

    """
    if np.random.random() > epsilon:
        return np.argmax(q_values)

    return action_space.sample()


def linear_interpolation(left: float, right: float, alpha: float) -> float:
    """ linear interpolation between l and r with alpha

    Args:
         left: (`float`):
         right: (`float`):
         alpha: (`float`):

    RETURNS (`float`):

    """
    return left + alpha * (right - left)


class PiecewiseSchedule():
    """ scheduler advancing piecewise """

    def __init__(
            self,
            endpoints: List[tuple],
            interpolation=linear_interpolation,  # FIXME: add type annotation
            outside_value=None):  # FIXME: add type annotation
        """ Piecewise Schedule

        Args:
             endpoints: (`List[tuple]`) [(int, int)]:

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
        assert self._outside_value is not None
        return self._outside_value


def sample_n_unique(sampling_f: callable, num: int) -> list:
    """ samples n **unique** instances using

    Args:
         sampling_f: (`callable`): the sampling function (is called to sample)
         num: (`int`): number of **unique** samples

    RETURNS (`list`): a list of samples

    """

    res = []
    while len(res) < num:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def sample_n(sampling_f: callable, num: int) -> list:
    """ samples n **non-unique** instances

    Assumes: sampling_f() can be called and returns comparable objects
    Note: can return duplicates

    Args:
         sampling_f: (`callable`): the function to call to sample
         num: (`int`): the amount of samples

    RETURNS (`list`): list of samples

    """

    res = []
    while len(res) < num:
        res.append(sampling_f())
    return res


# please, for the love of everything good in this world, don't refer to
# this directly
____SESS = None


def tf_init():
    """init initiates the wrapper (called once)

    anything done with TF before calling this function is bogus
    """

    global ____SESS  # pylint: disable=global-statement
    assert ____SESS is None, "Please initiate tf_wrapper only once"

    tf.reset_default_graph()

    tf_config = tf.ConfigProto(
        device_count={'GPU': 0},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
    )

    ____SESS = tf.Session(config=tf_config)


def tf_get_session():
    """ returns current session """

    global ____SESS  # pylint: disable=global-statement

    if ____SESS is None:
        tf_init()

    return ____SESS


class DiscreteSpace():
    """ DiscreteSpace discrete uninterupted space of some shape """

    def __init__(self, dim: List[int]):
        """ initiates a discrete space of size dim

        Args:
             dim: (`List[int]`): is a list of dimensions

        """
        assert isinstance(dim, list)

        self.dim = np.array(dim)
        self.n = np.prod(self.dim)
        self.shape = self.dim.shape

    def dimensions(self) -> np.array:
        """ returns the range of each dimension

        RETURNS (`np.array`): each member is the size of its dimension

        """
        return self.dim

    def sample(self) -> np.array:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return (np.random.random(self.shape) * self.dim).astype(int)
