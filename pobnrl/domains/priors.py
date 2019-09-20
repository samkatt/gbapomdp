""" priors over the domains """

from typing import Tuple, Set
import abc
import random

from numpy.random import dirichlet

from domains.collision_avoidance import CollisionAvoidance
from domains.gridworld import GridWorld
from domains.tiger import Tiger
from environments import Simulator, EncodeType


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
        """ initiate the prior, will make observation one-hot encoded

        Args:
             encoding: (`pobnrl.environments.EncodeType`):

        """

        self._encoding = encoding

    def sample(self) -> Simulator:
        """ returns a Tiger instance with some correct observation prob

        This prior over the observation probability is a Dirichlet with alpha
        [6,4]

        RETURNS (`pobnrl.environments.Simulator`):

        """
        sampled_observation_probs = [dirichlet([6, 4])[0], dirichlet([6, 4])[0]]

        return Tiger(encoding=self._encoding, correct_obs_probs=sampled_observation_probs)


class GridWorldPrior(Prior):
    """ a prior that returns gridworlds without slow cells

    The slow cells are sampled with 1/3 chance, meaning that each location has
    a .333 chance of being a slow cell

    """

    def __init__(self, size: int, encoding: EncodeType):
        """ creates a prior for the `gridworld` of size and with `encoding`

        Args:
             size: (`int`):
             encoding: (`pobnrl.environments.EncodeType`):

        """

        self._grid_size = size
        self._encoding = encoding

    def sample(self) -> Simulator:
        """  samples a `pobnrl.domains.gridworld.GridWorld` Gridworld of given size and encoding with a random set of slow cells """

        slow_cells: Set[Tuple[int, int]] = set()

        for i in range(self._grid_size):
            for j in range(self._grid_size):
                if random.random() < 1 / 3:
                    slow_cells.add((i, j))

        return GridWorld(self._grid_size, self._encoding, slow_cells)


class CollisionAvoidancePrior(Prior):
    """ a prior that returns collision avoidance with various obstacle behaviours

    The obstacle behaviour (accross all states) is sampled uniformly

    """

    def __init__(self, size: int):
        """ creates a `pobnrl.domains.collision_avoidance.CollisionAvoidance` prior of `size`

        Args:
             size: (`int`):
        """

        self._size = size

    def sample(self) -> Simulator:
        """  returns `pobnrl.domains.collision_avoidance.CollisionAvoidance` with random obstacle behavior """

        sampled_behaviour = tuple(dirichlet([.5, .9, .5]))

        return CollisionAvoidance(self._size, sampled_behaviour)  # type: ignore
