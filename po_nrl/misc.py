""" miscellaneous functions """

from enum import Enum
from typing import List, Union
import abc
import logging
import random

import numpy as np

import po_nrl.pytorch_api


def set_random_seed(seed: int) -> None:
    """ sets the random seed of our program

    NOTE that this function is not designed to be able to replicate
    experiments, this would require more code. This is merely to **ensure
    experiments are different**. Sometimes you will want to run scripts
    multiple times to then later aggregate the results: if programs use the
    current time as random seed then all runs that are started at the same time
    will result in the similar behaviour. This is to circumvent that.

    Sets `numpy` and `random` seed.


    Args:
         seed: (`int`): what seed to set it to

    RETURNS (`None`):

    """
    np.random.seed(seed)
    random.seed(seed)
    po_nrl.pytorch_api.set_random_seed(seed)


class POBNRLogger:
    """ logger, inherit in order to use logging function with self.log() """

    class LogLevel(Enum):
        """ log levels """
        V0 = 1000  # NO messages
        V1 = 30    # print results and setup
        V2 = 20    # print episodes
        V3 = 15    # print episode level agent things
        V4 = 10    # print time steps level agent things
        V5 = 5     # hardcore debugging

        @staticmethod
        def create(level: int) -> 'POBNRLogger.LogLevel':
            """ creates a loglevel from string

            Args:
                 level: (`int`): in [0 ... 5]

            RETURNS (`po_nrl.misc.POBNRLogger.LogLevel`):

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
                "[%(asctime)s] %(message)s",
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


class Space(abc.ABC):
    """ some mathematical space """

    @abc.abstractproperty
    def ndim(self) -> int:
        """ returns number of dimensions of the space """

    @abc.abstractmethod
    def sample(self) -> np.ndarray:
        """ samples from the space """

    @abc.abstractmethod
    def contains(self, elem: np.ndarray) -> bool:
        """ returns whether `elem` is in this space """


class DiscreteSpace(Space):
    """ DiscreteSpace discrete uninterupted space of some shape """

    def __init__(self, size: Union[List[int], np.ndarray]):
        """ initiates a discrete space of size size

        Args:
             size: (`Union[List[int], np.ndarray]`): is a list of dimension ranges

        """

        self.size = np.array(size).astype(int)
        self.num_elements: int = np.prod(self.size)
        self._indexing_steps = np.array([np.prod(self.size[:i]) for i in range(len(self.size))]).astype(int)
        self.dim_cumsum = np.concatenate([[0], np.cumsum(self.size)])

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

    def sample(self) -> np.ndarray:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return (np.random.random(self.ndim) * self.size).astype(int)  # pylint: disable=no-member

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
