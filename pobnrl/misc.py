""" miscellaneous functions

Contains:
    * sampling operations
    * mathematical spaces or sets
    * wrapper functions for interaction with tensorflow
    * (exploration) schedulars
    * e-greedy methods
    * logger class

"""

from contextlib import contextmanager
from enum import Enum
from typing import List, Union, Any
import logging
import os

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


class POBNRLogger:
    """ logger, inherit in order to use logging function with self.log() """

    class LogLevel(Enum):
        """ log levels """
        V0 = 1000  # NO messages
        V1 = 30    # print results
        V2 = 20    # print episodes
        V3 = 15    # print episode level agent things
        V4 = 10    # print time steps level agent things
        V5 = 5     # hardcore debugging

        @staticmethod
        def create(level: int) -> 'POBNRLogger.LogLevel':
            """ creates a loglevel from string

            Args:
                 level: (`int`): in [0 ... 5]

            RETURNS (`pobnrl.misc.POBNRLogger.LogLevel`):

            """

            return POBNRLogger.LogLevel['V' + str(level)]

    _level = LogLevel.V0
    registered_loggers: List['POBNRLogger'] = []

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

    def __init__(self, name: str = ""):
        """ creates a logger """

        if not name:
            name = self.__class__.__name__

        self.logger = logging.Logger(name)

        self.logger.setLevel(POBNRLogger._level.value)

        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                f"[%(asctime)s] %(message)s",
                "%H:%M"
            )
        )
        self.logger.addHandler(handler)

        self._enabled = True

        POBNRLogger.registered_loggers.append(self)

    def log(self, lvl: LogLevel, msg: str):
        """ logs message """

        if self._enabled:
            self.logger.log(lvl.value, lvl.name + ": " + msg)

    def disable_logging(self):
        """ disable logger """
        self._enabled = False

    @classmethod
    def log_is_on(cls, lvl: LogLevel) -> bool:
        """ returns whether logging is on for given level

        Args:
             lvl: (`LogLevel`): level to check

        RETURNS (`bool`): True if messages with this log level would be printed

        """

        return lvl.value >= cls._level.value


# please, for the love of everything good in this world, don't refer to this
_SESS: tf.Session = None


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

    logger.log(POBNRLogger.LogLevel.V1, "initiating tensorflow session")

    tf_config = tf.ConfigProto(
        device_count={'GPU': int(use_gpu)},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid print statements
    _SESS = tf.Session(config=tf_config)

    yield _SESS

    # __exit__()

    logger.log(POBNRLogger.LogLevel.V1, "closing tensorflow session")

    _SESS.close()
    _SESS = None

    tf.reset_default_graph()


def tf_run(operations, **kwargs) -> Any:
    """ runs a tf session """
    return _SESS.run(operations, **kwargs)


class DiscreteSpace():
    """ DiscreteSpace discrete uninterupted space of some shape """

    def __init__(self, size: Union[List[int], np.ndarray]):
        """ initiates a discrete space of size size

        Args:
             size: (`List[int]`): is a list of dimension ranges

        """

        self.size = np.array(size).astype(int)
        self.num_elements: int = np.prod(self.size)
        self._indexing_steps = np.array([np.prod(self.size[:i]) for i in range(len(self.size))]).astype(int)

    @property
    def n(self) -> int:  # pylint: disable=invalid-name
        """ Number of elements in space

        While the naming is pretty awful, it is consistent with the `Space`
        class of open AI gym, which I prioritized here

        RETURNS (`int`):

        """
        return self.num_elements

    @property
    def ndim(self) -> int:
        """ returns the numbe of dimensions

        RETURNS (`int`): number of dimensions

        """
        return len(self.size)

    def contains(self, elem: np.ndarray) -> bool:
        """ returns `this` contains `elem`

        Args:
             elem: (`np.ndarray`): element to check against

        RETURNS (`bool`):

        """

        return elem.shape == (self.ndim,) and \
            (elem >= 0).all() and (elem < self.size).all()

    def sample(self) -> np.array:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return (np.random.random(self.ndim) * self.size).astype(int)

    def index_of(self, elem: np.ndarray) -> int:
        """ returns the index of an element (projects to single dimension)

        Args:
             elem: (`np.ndarray`): the element to project

        RETURNS (`int`): projection

        """
        assert self.contains(elem)
        return np.dot(elem, self._indexing_steps)

    def __repr__(self):
        return f"DiscreteSpace of size {self.size}"
