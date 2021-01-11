""" tiger environment """

from typing import List, Optional
import numpy as np

from po_nrl.environments import Environment, EnvironmentInteraction, ActionSpace
from po_nrl.environments import Simulator, SimulationResult, EncodeType
from po_nrl.misc import DiscreteSpace, POBNRLogger


class Tiger(Environment, Simulator, POBNRLogger):
    """ the tiger environment """

    # consts
    LEFT = 0
    LISTEN = 2

    GOOD_DOOR_REWARD = 10
    BAD_DOOR_REWARD = -100

    LISTEN_REWARD = -1

    ELEM_TO_STRING = ["L", "R"]

    def __init__(self, encoding: EncodeType, correct_obs_probs: Optional[List[float]] = None):
        """  construct the tiger environment

        Args:
             encoding_type: (`EncodeType`):
             correct_obs_probs: (`Optional[List[float]]`):

        """

        if not correct_obs_probs:
            correct_obs_probs = [.85, .85]

        assert 0 <= correct_obs_probs[0] <= 1, \
            f"observation prob {correct_obs_probs[0]} not a probability"
        assert 0 <= correct_obs_probs[1] <= 1, \
            f"observation prob {correct_obs_probs[1]} not a probability"

        POBNRLogger.__init__(self)

        self._correct_obs_probs = correct_obs_probs

        self._use_one_hot_obs = encoding == EncodeType.ONE_HOT

        self._state_space = DiscreteSpace([2])
        self._action_space = ActionSpace(3)
        self._obs_space = DiscreteSpace([2, 2]) if self._use_one_hot_obs else DiscreteSpace([3])

        self._state = self.sample_start_state()

    @property
    def state(self):
        """ returns current state """
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """ sets state

        Args:
             state: (`np.ndarray`): [0] or [1]

        """

        assert state.shape == (1,), f"{state} not correct shape"
        assert 2 > state[0] >= 0, f"{state} not valid"

        self._state = state

    @property
    def state_space(self) -> DiscreteSpace:
        """ a `po_nrl.misc.DiscreteSpace`([2]) space """
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        """ a `po_nrl.environments.ActionSpace`([3]) space """
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `po_nrl.misc.DiscreteSpace`([1,1]) space if one-hot, otherwise [3]"""
        return self._obs_space

    def encode_observation(self, observation: int) -> np.ndarray:
        """ encodes the observation for usage outside of `this`

        This wraps the `int` observation into a numpy array. Either directly,
        or with one-hot encoding if `Tiger` was initiated with that parameter
        to true

        Args:
             observation: (`int`): 0, 1= hear behind door, 2=null

        RETURNS (`np.ndarray`):

        """

        if not self._use_one_hot_obs:
            return np.array([observation])

        # use one hot encoding
        obs = np.ones(2)

        # not left or right means [1,1] observation (basically a 'null')
        if observation > 1:
            return obs

        obs[int(not observation)] = 0

        return obs

    @staticmethod
    def sample_start_state() -> np.ndarray:
        """ samples a random state (tiger left or right)

        RETURNS (`np.narray`): an initial state (in [[0],[1]])

        """
        return np.array([np.random.randint(0, 2)])

    def sample_observation(self, loc: int, listening: bool) -> int:
        """ samples an observation, listening stores whether agent is listening

        Args:
             loc: (`int`): 0 is tiger left, 1 is tiger right
             listening: (`bool`): whether the agent is listening

        RETURNS (`int`): the observation: 0 = left, 1 = right, 2 = null

        """

        if not listening:
            return self.LISTEN

        return loc if np.random.random() < self._correct_obs_probs[loc] else int(not loc)  # pylint: disable=no-member

    def reset(self) -> np.ndarray:
        """ Resets internal state and return first observation

        Resets the internal state randomly ([0] or [1])
        Returns [1,1] as a 'null' initial observation

        """
        self._state = self.sample_start_state()
        return self.encode_observation(self.LISTEN)

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """ Simulates stepping from state using action. Returns interaction

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             state: (`np.ndarray`): [0] is tiger left, [1] is tiger right
             action: (`int`): 0 is open left, 1 is open right or 2 is listen

        RETURNS (`po_nrl.environments.SimulationResult`): the transition

        """

        if action != self.LISTEN:
            obs = self.sample_observation(state[0], False)
            new_state = self.sample_start_state()

        else:  # not opening door
            obs = self.sample_observation(state[0], True)
            new_state = state.copy()

        return SimulationResult(new_state, self.encode_observation(obs))

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """ A constant if listening, penalty if opening to door, and reward otherwise

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
        """ True if opening a door

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

    def step(self, action: int) -> EnvironmentInteraction:
        """ performs a step in the tiger environment given action

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             action: (`int`): 0 is open left, 1 is open right or 2 is listen

        RETURNS (`po_nrl.environments.EnvironmentInteraction`): the transition

        """

        sim_result = self.simulation_step(self.state, action)
        reward = self.reward(self.state, action, sim_result.state)
        terminal = self.terminal(self.state, action, sim_result.state)

        if self.log_is_on(POBNRLogger.LogLevel.V2):
            if action == self.LISTEN:
                descr = "the agent hears " \
                    + self.ELEM_TO_STRING[self.obs2index(sim_result.observation)]
            else:  # agent is opening door
                descr = f"the agent opens {self.ELEM_TO_STRING[action]} ({reward})"

            self.log(
                POBNRLogger.LogLevel.V2,
                f"With tiger {self.ELEM_TO_STRING[self.state[0]]}, {descr}"
            )

        self.state = sim_result.state

        return EnvironmentInteraction(sim_result.observation, reward, terminal)

    def obs2index(self, observation: np.ndarray) -> int:
        """ projects the observation as an int

        Args:
             observation: (`np.ndarray`): observation to project

        RETURNS (`int`): int representation of observation

        """

        assert self.observation_space.contains(observation), f"{observation} not in space"

        if not self._use_one_hot_obs:
            return observation[0]

        return int(self._obs_space.index_of(observation) - 1)

    def __repr__(self) -> str:
        encoding_descr = "one_hot" if self._use_one_hot_obs else "default"
        return f"Tiger problem ({encoding_descr} encoding) with obs prob {self._correct_obs_probs}"
