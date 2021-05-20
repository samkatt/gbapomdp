"""The tiger problem implemented as domain"""

from logging import Logger
from typing import List, Optional

import numpy as np

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
)
from general_bayes_adaptive_pomdps.domains import domain
from general_bayes_adaptive_pomdps.misc import DiscreteSpace, LogLevel


class Tiger(domain.Domain):
    """The actual domain"""

    # consts
    LEFT = 0
    LISTEN = 2

    GOOD_DOOR_REWARD = 10
    BAD_DOOR_REWARD = -100

    LISTEN_REWARD = -1

    ELEM_TO_STRING = ["L", "R"]

    def __init__(
        self,
        one_hot_encode_observation: bool,
        correct_obs_probs: Optional[List[float]] = None,
    ):
        """Construct the tiger domain

        Args:
             one_hot_encode_observation: (`bool`):
             correct_obs_probs: (`Optional[List[float]]`):

        """

        if not correct_obs_probs:
            correct_obs_probs = [0.85, 0.85]

        assert (
            0 <= correct_obs_probs[0] <= 1
        ), f"observation prob {correct_obs_probs[0]} not a probability"
        assert (
            0 <= correct_obs_probs[1] <= 1
        ), f"observation prob {correct_obs_probs[1]} not a probability"

        self._logger = Logger(self.__class__.__name__)

        self._correct_obs_probs = correct_obs_probs

        self._use_one_hot_obs = one_hot_encode_observation

        self._state_space = DiscreteSpace([2])
        self._action_space = ActionSpace(3)
        self._obs_space = (
            DiscreteSpace([2, 2]) if self._use_one_hot_obs else DiscreteSpace([3])
        )

        self._state = self.sample_start_state()

    @property
    def state(self):
        """ returns current state """
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """sets state

        Args:
             state: (`np.ndarray`): [0] or [1]

        """

        assert state.shape == (1,), f"{state} not correct shape"
        assert 2 > state[0] >= 0, f"{state} not valid"

        self._state = state

    @property
    def state_space(self) -> DiscreteSpace:
        """ a `general_bayes_adaptive_pomdps.misc.DiscreteSpace` ([2]) space """
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        """ a `general_bayes_adaptive_pomdps.core.ActionSpace` ([3]) space """
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `general_bayes_adaptive_pomdps.misc.DiscreteSpace` ([1,1]) space if one-hot, otherwise [3]"""
        return self._obs_space

    def encode_observation(self, observation: int) -> np.ndarray:
        """encodes the observation for usage outside of `this`

        This wraps the `int` observation into a numpy array. Either directly,
        or with one-hot encoding if `Tiger` was initiated with that parameter
        to true

        Args:
             observation: (`int`): 0, 1= hear behind door, 2=null

        RETURNS (`np.ndarray`):

        """

        if not self._use_one_hot_obs:
            return np.array([observation], dtype=int)

        # use one hot encoding
        obs = np.ones(2, dtype=int)

        # not left or right means [1,1] observation (basically a 'null')
        if observation > 1:
            return obs

        obs[int(not observation)] = 0

        return obs

    @staticmethod
    def sample_start_state() -> np.ndarray:
        """samples a random state (tiger left or right)

        RETURNS (`np.narray`): an initial state (in [[0],[1]])

        """
        return np.array([np.random.randint(0, 2)], dtype=int)

    def sample_observation(self, loc: int, listening: bool) -> int:
        """samples an observation, listening stores whether agent is listening

        Args:
             loc: (`int`): 0 is tiger left, 1 is tiger right
             listening: (`bool`): whether the agent is listening

        RETURNS (`int`): the observation: 0 = left, 1 = right, 2 = null

        """

        if not listening:
            return self.LISTEN

        return (
            loc if np.random.random() < self._correct_obs_probs[loc] else int(not loc)
        )

    def reset(self) -> np.ndarray:
        """Resets internal state and return first observation

        Resets the internal state randomly ([0] or [1])
        Returns [1,1] as a 'null' initial observation

        """
        self._state = self.sample_start_state()
        return self.encode_observation(self.LISTEN)

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """Simulates stepping from state using action. Returns interaction

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             state: (`np.ndarray`): [0] is tiger left, [1] is tiger right
             action: (`int`): 0 is open left, 1 is open right or 2 is listen

        RETURNS (`general_bayes_adaptive_pomdps.core.SimulationResult`): the transition

        """

        if action != self.LISTEN:
            obs = self.sample_observation(state[0], False)
            new_state = self.sample_start_state()

        else:  # not opening door
            obs = self.sample_observation(state[0], True)
            new_state = state.copy()

        return SimulationResult(new_state, self.encode_observation(obs))

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """A constant if listening, penalty if opening to door, and reward otherwise

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`):

        """

        assert self.state_space.contains(state), f"{state} not in space"
        assert self.state_space.contains(new_state), f"{new_state} not in space"
        assert self.action_space.contains(action), f"{action} not in space"

        if action == self.LISTEN:
            return self.LISTEN_REWARD

        return self.GOOD_DOOR_REWARD if action == state[0] else self.BAD_DOOR_REWARD

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if opening a door

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`):

        """

        assert self.state_space.contains(state), f"{state} not in space"
        assert self.state_space.contains(new_state), f"{new_state} not in space"
        assert self.action_space.contains(action), f"{action} not in space"

        return bool(action != self.LISTEN)

    def step(self, action: int) -> DomainStepResult:
        """Performs a step in the tiger problem given action

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             action: (`int`): 0 is open left, 1 is open right or 2 is listen

        RETURNS (`general_bayes_adaptive_pomdps.core.EnvironmentInteraction`): the transition

        """

        sim_result = self.simulation_step(self.state, action)
        reward = self.reward(self.state, action, sim_result.state)
        terminal = self.terminal(self.state, action, sim_result.state)

        if self._logger.isEnabledFor(LogLevel.V2.value):
            if action == self.LISTEN:
                descr = (
                    "the agent hears "
                    + self.ELEM_TO_STRING[self.obs2index(sim_result.observation)]
                )
            else:  # agent is opening door
                descr = f"the agent opens {self.ELEM_TO_STRING[action]} ({reward})"

            self._logger.log(
                LogLevel.V2.value,
                f"With tiger {self.ELEM_TO_STRING[self.state[0]]}, {descr}",
            )

        self.state = sim_result.state

        return DomainStepResult(sim_result.observation, reward, terminal)

    def obs2index(self, observation: np.ndarray) -> int:
        """projects the observation as an int

        Args:
             observation: (`np.ndarray`): observation to project

        RETURNS (`int`): int representation of observation

        """

        assert self.observation_space.contains(
            observation
        ), f"{observation} not in space"

        if not self._use_one_hot_obs:
            return observation[0]

        return int(self._obs_space.index_of(observation) - 1)

    def __repr__(self) -> str:
        encoding_descr = "one_hot" if self._use_one_hot_obs else "default"
        return f"Tiger problem ({encoding_descr} encoding) with obs prob {self._correct_obs_probs}"


class TigerPrior(domain.DomainPrior):
    """standard prior over the tiger domain

    The transition model is known, however the probability of observing the
    tiger correctly is not. Here we assume a `Dir(prior * total_counts
    ,(1-prior) * total_counts)` belief over this distribution.

    `prior` is computed by the `prior_correctness`: 1 -> .85, whereas 0 ->
    .625, linear mapping in between

    """

    def __init__(
        self,
        num_total_counts: float,
        prior_correctness: float,
        one_hot_encode_observation: bool,
    ):
        """initiate the prior, will make observation one-hot encoded

        Args:
             num_total_counts: (`float`): Number of total counts of Dir prior
             prior_correctness: (`float`): How correct the observation model is: [0, 1] -> [.625, .85]
             one_hot_encode_observation: (`bool`):

        """

        if num_total_counts <= 0:
            raise ValueError(
                f"Assume positive number of total counts, not {num_total_counts}"
            )

        if not 0 <= prior_correctness <= 1:
            raise ValueError(
                f"`prior_correctness` must be [0,1], not {prior_correctness}"
            )

        # Linear mapping: [0, 1] -> [.625, .85]
        self._observation_prob = 0.625 + (prior_correctness * 0.225)
        self._total_counts = num_total_counts
        self._one_hot_encode_observation = one_hot_encode_observation

    def sample(self) -> domain.Domain:
        """returns a Tiger instance with some correct observation prob

        This prior over the observation probability is a Dirichlet with total
        counts and observation probability as defined during the initialization

        RETURNS (`general_bayes_adaptive_pomdps.core.Domain`):

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
            one_hot_encode_observation=self._one_hot_encode_observation,
            correct_obs_probs=sampled_observation_probs,
        )
