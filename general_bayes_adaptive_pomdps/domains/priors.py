""" priors over the domains """

from typing import Tuple, Set
import abc
import random

import numpy as np

from general_bayes_adaptive_pomdps.domains.collision_avoidance import CollisionAvoidance
from general_bayes_adaptive_pomdps.domains.gridworld import GridWorld
from general_bayes_adaptive_pomdps.domains.tiger import Tiger
from general_bayes_adaptive_pomdps.domains.road_racer import RoadRacer
from general_bayes_adaptive_pomdps.environments import Simulator, EncodeType


class Prior(abc.ABC):
    """ the interface to priors """

    @abc.abstractmethod
    def sample(self) -> Simulator:
        """sample a simulator

        RETURNS (`general_bayes_adaptive_pomdps.environments.Simulator`):
        """


class TigerPrior(Prior):
    """standard prior over the tiger domain

    The transition model is known, however the probability of observing the
    tiger correctly is not. Here we assume a `Dir(prior * total_counts
    ,(1-prior) * total_counts)` belief over this distribution.

    `prior` is computed by the `prior_correctness`: 1 -> .85, whereas 0 ->
    .625, linear mapping in between

    """

    def __init__(
        self, num_total_counts: float, prior_correctness, encoding: EncodeType
    ):
        """initiate the prior, will make observation one-hot encoded

        Args:
             num_total_counts: (`float`): Number of total counts of Dir prior
             prior_correctness: (`float`): How correct the observation model is: [0, 1] -> [.625, .85]
             encoding: (`general_bayes_adaptive_pomdps.environments.EncodeType`):

        """

        if num_total_counts <= 0:
            raise ValueError(
                f"Assume positive number of total counts, not {num_total_counts}"
            )

        if not 0 <= prior_correctness < 1:
            raise ValueError(
                f"`prior_correctness` must be [0,1], not {prior_correctness}"
            )

        # Linear mapping: [0, 1] -> [.625, .85]
        self._observation_prob = 0.625 + (prior_correctness * 0.225)
        self._total_counts = num_total_counts
        self._encoding = encoding

    def sample(self) -> Simulator:
        """returns a Tiger instance with some correct observation prob

        This prior over the observation probability is a Dirichlet with total
        counts and observation probability as defined during the initialization

        RETURNS (`general_bayes_adaptive_pomdps.environments.Simulator`):

        """
        sampled_observation_probs = [
            np.random.dirichlet(
                [
                    self._observation_prob * self._total_counts,
                    (1 - self._observation_prob) * self._total_counts,
                ]
            )[0],
            np.random.dirichlet(
                [
                    self._observation_prob * self._total_counts,
                    (1 - self._observation_prob) * self._total_counts,
                ]
            )[0],
        ]

        return Tiger(
            encoding=self._encoding, correct_obs_probs=sampled_observation_probs
        )


class GridWorldPrior(Prior):
    """a prior that returns gridworlds without slow cells

    The slow cells are sampled with 1/3 chance, meaning that each location has
    a .333 chance of being a slow cell

    """

    def __init__(self, size: int, encoding: EncodeType):
        """creates a prior for the `gridworld` of size and with `encoding`

        Args:
             size: (`int`):
             encoding: (`general_bayes_adaptive_pomdps.environments.EncodeType`):

        """

        self._grid_size = size
        self._encoding = encoding

    def sample(self) -> Simulator:
        """samples a `general_bayes_adaptive_pomdps.domains.gridworld.GridWorld`

        Gridworld is of given size and encoding with a random set of slow cells
        """

        slow_cells: Set[Tuple[int, int]] = set()

        for i in range(self._grid_size):
            for j in range(self._grid_size):
                if random.random() < 1 / 3:
                    slow_cells.add((i, j))

        return GridWorld(self._grid_size, self._encoding, slow_cells)


class CollisionAvoidancePrior(Prior):
    """a prior that returns collision avoidance with various obstacle behaviours

    The obstacle behaviour (accross all states) is sampled uniformly

    """

    def __init__(self, size: int, num_total_counts: float):
        """creates a `general_bayes_adaptive_pomdps.domains.collision_avoidance.CollisionAvoidance` prior of `size`

        Args:
             size: (`int`):
             num_total_counts: (`float`):
        """

        if num_total_counts <= 0:
            raise ValueError("Assume positive number of total counts")

        if size <= 0:
            raise ValueError("Assume positive grid size")

        self._size = size
        self._num_total_counts = num_total_counts

    def sample(self) -> Simulator:
        """returns `general_bayes_adaptive_pomdps.domains.collision_avoidance.CollisionAvoidance`

        Domain has with random obstacle behavior
        """

        sampled_behaviour = tuple(
            np.random.dirichlet(np.array([0.05, 0.9, 0.05]) * self._num_total_counts)
        )

        return CollisionAvoidance(self._size, sampled_behaviour)  # type: ignore


class RoadRacerPrior(Prior):
    """standard prior over the road racer domain

    The agent's transition and observation model is known, however the other
    cars' speed is not. We assign a expected model of p=.5 for advancing to
    each lane.

    """

    def __init__(self, num_lanes: int, num_total_counts: float):
        """initiate the prior, will make observation one-hot encoded

        Args:
             num_lanes: (`float`): number of lanes in the domain
             num_total_counts: (`float`): number of total counts of Dir prior

        """

        if num_total_counts <= 0:
            raise ValueError("Assume positive number of total counts")

        self._total_counts = num_total_counts
        self._num_lanes = num_lanes

    def sample(self) -> Simulator:
        """returns a Road Racer instance with some sampled set of lane speeds

        The prior over each lane advancement probability  is .5

        RETURNS (`general_bayes_adaptive_pomdps.environments.Simulator`):

        """
        sampled_lane_speeds = np.random.beta(
            0.5 * self._total_counts, 0.5 * self._total_counts, self._num_lanes
        )

        return RoadRacer(sampled_lane_speeds)
