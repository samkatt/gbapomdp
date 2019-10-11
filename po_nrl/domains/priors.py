""" priors over the domains """

from typing import Tuple, Set
import abc
import random

import numpy as np

from po_nrl.domains.collision_avoidance import CollisionAvoidance
from po_nrl.domains.gridworld import GridWorld
from po_nrl.domains.tiger import Tiger
from po_nrl.environments import Simulator, EncodeType


class Prior(abc.ABC):
    """ the interface to priors """

    @abc.abstractmethod
    def sample(self) -> Simulator:
        """ sample a simulator

        RETURNS (`po_nrl.environments.Simulator`):
        """


class TigerPrior(Prior):
    """ standard prior over the tiger domain

    The transition model is known, however the probability of observing the
    tiger correctly is not. Here we assume a Dir(.6 * total_counts ,.4 * total_counts) belief over this
    distribution.

    """

    def __init__(self, num_total_counts: float, encoding: EncodeType):
        """ initiate the prior, will make observation one-hot encoded

        Args:
             num_total_counts: (`float`): number of total counts of Dir prior
             encoding: (`po_nrl.environments.EncodeType`):

        """

        if num_total_counts <= 0:
            raise ValueError('Assume positive number of total counts')

        self._total_counts = num_total_counts
        self._encoding = encoding

    def sample(self) -> Simulator:
        """ returns a Tiger instance with some correct observation prob

        This prior over the observation probability is a Dirichlet with alpha
        [6,4]

        RETURNS (`po_nrl.environments.Simulator`):

        """
        sampled_observation_probs = [
            np.random.dirichlet([.6 * self._total_counts, .4 * self._total_counts])[0],
            np.random.dirichlet([.6 * self._total_counts, .4 * self._total_counts])[0]
        ]

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
             encoding: (`po_nrl.environments.EncodeType`):

        """

        self._grid_size = size
        self._encoding = encoding

    def sample(self) -> Simulator:
        """  samples a `po_nrl.domains.gridworld.GridWorld` Gridworld of given size and encoding with a random set of slow cells """

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

    def __init__(self, size: int, num_total_counts: float):
        """ creates a `po_nrl.domains.collision_avoidance.CollisionAvoidance` prior of `size`

        Args:
             size: (`int`):
             num_total_counts: (`float`):
        """

        if num_total_counts <= 0:
            raise ValueError('Assume positive number of total counts')

        if size <= 0:
            raise ValueError('Assume positive grid size')

        self._size = size
        self._num_total_counts = num_total_counts

    def sample(self) -> Simulator:
        """  returns `po_nrl.domains.collision_avoidance.CollisionAvoidance` with random obstacle behavior """

        sampled_behaviour = tuple(np.random.dirichlet(np.array([.05, .9, .05]) * self._num_total_counts))

        return CollisionAvoidance(self._size, sampled_behaviour)  # type: ignore
