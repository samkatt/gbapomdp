""" priors over the domains """

from typing import Tuple, Set
import abc
import random

from numpy.random import dirichlet

from environments import Simulator, EncodeType

from .tiger import Tiger
from .gridworld import GridWorld


class Prior(abc.ABC):
    """ the interface to priors """

    @abc.abstractmethod
    def sample(self) -> Simulator:
        """ sample a simulator

        RETURNS (`pobnrl.environments.Simulator`):
        """


class TigerPrior(Prior):
    """ standard prior over the tiger domain

    The transition model is known, however the probability of observing the
    tiger correctly is not. Here we assume a Dir(6,4) belief over this
    distribution.

    """

    def __init__(self, encoding: EncodeType):
        """ initiate the prior, will make observation one-hot encoded"""

        self._encoding = encoding

    def sample(self) -> Simulator:
        """ returns a Tiger instance with some correct observation prob

        This prior over the observation probability is a Dirichlet with alpha
        [6,4]

        RETURNS (`Simulator`):

        """
        sampled_observation_probs = [dirichlet([6, 4])[0], dirichlet([6, 4])[0]]

        return Tiger(encoding=self._encoding, correct_obs_probs=sampled_observation_probs)


class GridWorldPrior(Prior):
    """ a prior that returns gridworlds without slow cells """

    def __init__(self, size: int, encoding: EncodeType):
        """ creates a prior for the `gridworld` of size and with `encoding`

        Args:
             size: (`int`):
             encoding: (`EncodeType`):

        """

        self._grid_size = size
        self._encoding = encoding

    def sample(self) -> Simulator:
        """  returns Gridworld of given size and encoding with a random set of slow cells

        The slow cells are sampled uniformly, meaning that each location has a
        .5 chance of being a slow cell

        """

        slow_cells: Set[Tuple[int, int]] = set()

        for i in range(self._grid_size):
            for j in range(self._grid_size):
                if random.choice([True, False]):
                    slow_cells.add((i, j))

        return GridWorld(self._grid_size, self._encoding, slow_cells)
