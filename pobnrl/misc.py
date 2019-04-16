""" miscellaneous functions

Contains:
    * sampling operations
    * mathematical spaces or sets
    * wrapper functions for interaction with tensorflow:
    * (exploration) schedulars
    * e-greedy methods

"""

from contextlib import contextmanager
from typing import List
import os

import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# TODO: change to log function that wraps logger
log_level = {"spam": 5, "debug": 10, "verbose": 15, "info": 20}

# please, for the love of everything good in this world, don't refer to this
_SESS = None


@contextmanager
def tf_session(use_gpu: bool):
    """ used as context to run TF in

    e.g.:
    with tf_session():

        ...
        tf_run(...)


    Args:
         use_gpu: (`bool`): whether to use gpus
    """

    # __enter__
    global _SESS  # pylint: disable=global-statement
    assert _SESS is None, "Please initiate tf_wrapper only once"

    logger.log(log_level['verbose'], "initiating tensorflow session")

    tf_config = tf.ConfigProto(
        device_count={'GPU': int(use_gpu)},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid print statements
    _SESS = tf.Session(config=tf_config)

    yield _SESS

    # __exit__()

    logger.log(log_level['verbose'], "closing tensorflow session")

    _SESS.close()
    _SESS = None

    tf.reset_default_graph()


def tf_run(operations, **kwargs):
    """ runs a tf session """
    return _SESS.run(operations, **kwargs)


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

    def __repr__(self):
        return f"DiscreteSpace of shape {self.dimensions}"
