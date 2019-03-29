""" miscellaneous functions

Contains:
    * sampling operations
    * mathematical spaces or sets
    * wrapper functions for interaction with tensorflow:
    * (exploration) schedulars
    * e-greedy methods

"""

from typing import List, Callable

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


class PiecewiseSchedule():  # pylint: disable=too-few-public-methods
    """ scheduler advancing piecewise """

    def __init__(
            self,
            endpoints: List[tuple],
            interpolation: Callable[[float, float, float], float] = linear_interpolation,
            outside_value: float = None):
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


# please, for the love of everything good in this world, don't refer to this
____SESS = None


def tf_init():
    """init initiates the wrapper (called once)

    anything done with TF before calling this function is bogus
    """

    global ____SESS  # pylint: disable=global-statement
    assert ____SESS is None, "Please initiate tf_wrapper only once"

    print("initiating tensorflow session")

    tf_config = tf.ConfigProto(
        device_count={'GPU': 0},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )

    ____SESS = tf.Session(config=tf_config)


def tf_close():
    """" closes the tf session

    Hopefully only called at the end of the program


    """

    global ____SESS  # pylint: disable=global-statement
    assert ____SESS is not None, "Cannot close session before initiating it"

    print("closing tensorflow session")

    ____SESS.close()


def tf_run(operations, **kwargs):
    """ runs a tf session """
    return tf_get_session().run(operations, **kwargs)


def tf_get_session():
    """ returns current session, please use sparingly and see `tf_run` """

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

        self._dim = np.array(dim)
        self.num_elements = np.prod(self._dim)
        self._shape = self._dim.shape

    @property
    def n(self) -> int:  # pylint: disable=invalid-name
        """ Number of elements in space

        While the naming is pretty awful, it is consistent with the `Space`
        class of open AI gym, which I prioritized here

        RETURNS (`int`):

        """
        return self.num_elements

    @property
    def dimensions(self) -> np.array:
        """ returns the range of each dimension

        RETURNS (`np.array`): each member is the size of its dimension

        """
        return self._dim

    @property
    def shape(self) -> tuple:
        """ returns the shape of the space

        Args:

        RETURNS (`tuple`): as like np.shape

        """
        return self._shape

    def sample(self) -> np.array:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return (np.random.random(self.shape) * self._dim).astype(int)
