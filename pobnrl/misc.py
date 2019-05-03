""" miscellaneous functions

Contains:
    * sampling operations
    * mathematical spaces or sets
    * wrapper functions for interaction with tensorflow:
    * (exploration) schedulars
    * e-greedy methods

"""

from contextlib import contextmanager
from enum import Enum
from typing import List, Tuple
import logging
import os

import numpy as np
import tensorflow as tf


class LogLevel(Enum):
    """ log levels """
    V0 = 1000
    V1 = 30
    V2 = 20
    V3 = 15
    V4 = 10
    V5 = 5

    @staticmethod
    def create(level: int) -> 'LogLevel':
        """ creates a loglevel from string

        Args:
             level: (`int`): in [0 ... 5]

        RETURNS (`pobnrl.misc.LogLevel`):

        """

        return LogLevel['V' + str(level)]


class POBNRLogger:
    """ logger, inherit in order to use logging function with self.log() """

    _level = LogLevel.V0
    registered_loggers = []

    @classmethod
    def set_level(cls, level: LogLevel):
        """ sets level of the loggers

        Anything that is logged with a **lower** level will be displayed

        Args:
             level: (`LogLevel`):

        """

        for logger in cls.registered_loggers:
            logger.logger.setLevel(level.value)

        cls._level = level

    def __init__(self, name: str):
        """ creates a logger with given name

        Args:
             name: (`str`): the name of the logger

        """

        self.logger = logging.Logger(name)

        self.logger.setLevel(POBNRLogger._level.value)

        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s: %(message)s \t\t\t(%(name)s)"
            )
        )
        self.logger.addHandler(handler)

        self._enabled = True

        POBNRLogger.registered_loggers.append(self)

    def log(self, lvl: LogLevel, msg: str):
        """ logs message """
        if self._enabled:
            self.logger.log(lvl.value, msg)

    def disable_logging(self):
        """ disable logger """
        self._enabled = False


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

    logger = POBNRLogger('tf_session')

    # __enter__
    global _SESS  # pylint: disable=global-statement
    assert _SESS is None, "Please initiate tf_wrapper only once"

    logger.log(LogLevel.V1, "initiating tensorflow session")

    tf_config = tf.ConfigProto(
        device_count={'GPU': int(use_gpu)},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid print statements
    _SESS = tf.Session(config=tf_config)

    yield _SESS

    # __exit__()

    logger.log(LogLevel.V1, "closing tensorflow session")

    _SESS.close()
    _SESS = None

    tf.reset_default_graph()


def tf_run(operations, **kwargs):
    """ runs a tf session """
    return _SESS.run(operations, **kwargs)


class DiscreteSpace():
    """ DiscreteSpace discrete uninterupted space of some shape

    TODO: contains()

    """

    def __init__(self, dim: List[int]):
        """ initiates a discrete space of size dim

        Args:
             dim: (`List[int]`): is a list of dimensions

        """

        # TODO: why..?
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

        TODO: rename to size?

        RETURNS (`np.array`): each member is the size of its dimension

        """
        return self._dim

    @property
    def shape(self) -> Tuple[int]:
        """ returns the shape of the space

        TODO: len, ndim?

        Args:

        RETURNS (`Tuple[int]`): as like np.shape

        """
        return self._shape

    def sample(self) -> np.array:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return (np.random.random(self.shape) * self._dim).astype(int)

    def __repr__(self):
        return f"DiscreteSpace of size {self.dimensions}"
