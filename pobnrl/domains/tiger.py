""" tiger environment """

from typing import List, Any
import copy
import numpy as np

from environments import Environment, EnvironmentInteraction, ActionSpace
from environments import POUCTSimulator, POUCTInteraction
from misc import DiscreteSpace, POBNRLogger, LogLevel


class Tiger(Environment, POUCTSimulator, POBNRLogger):
    """ the tiger environment """

    # consts
    LEFT = 0
    LISTEN = 2

    GOOD_DOOR_REWARD = 10
    BAD_DOOR_REWARD = -100

    LISTEN_REWARD = -1
    CORRECT_OBSERVATION_PROB = .85

    def __init__(self, verbose: bool):
        """ construct the tiger environment

        Args:
             verbose: (`bool`): whether to be verbose or not (print to stdout)

        """

        POBNRLogger.__init__(self)

        self._verbose = verbose
        self._state = self.sample_start_state()

        self._action_space = ActionSpace(3)
        self._obs_space = DiscreteSpace([2, 2])

        self._history: List[Any] = []

    @property
    def state(self):
        """ returns current state """
        return self._state

    @state.setter
    def state(self, state: int):
        """ sets state

        Args:
             state: (`int`):

        """

        assert 2 > state >= 0

        self._state = state

    @staticmethod
    def sample_start_state() -> int:
        """ samples a random state (tiger left or right)

        RETURNS (`int`): an initial state (in [0,1])

        """
        return np.random.randint(0, 2)

    def sample_observation(self, state: int, listening: bool) -> np.ndarray:
        """ samples an observation, listening stores whether agent is listening

        Args:
             state: (`int`): 0 is tiger left, 1 is tiger right
             listening: (`bool`): whether the agent is listening

        RETURNS (`np.array`): the observation (hot-encoded)

        """
        obs = np.zeros(self._obs_space.ndim)

        # not listening means [0,0] observation (basically a 'null')
        if not listening:
            return obs

        # 1-hot-encoding
        if np.random.random() < self.CORRECT_OBSERVATION_PROB:
            obs[state] = 1
        else:
            obs[int(not state)] = 1

        return obs

    def reset(self):
        """ resets internal state and return first observation

        resets the intenral state randomly (0 or 1)
        returns [0,0] as a 'null' initial observation

        """

        self._state = self.sample_start_state()

        if self.log_is_on(LogLevel.V2) and self._history:
            self.display_history()

        # record episodes every so often
        self._history = [copy.deepcopy(self.state)]

        return np.zeros(2)

    def simulation_step(self, state: int, action: int) -> POUCTInteraction:
        """ simulates stepping from state using action. Returns interaction

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             state: (`int`): 0 is tiger left, 1 is tiger right
             action: (`int`): 0 is open left, 1 is open right or 2 is listen

        RETURNS (`pobnrl.environments.POUCTInteraction`): the transition

        """

        if action != self.LISTEN:
            obs = self.sample_observation(state, False)
            terminal = True
            reward = self.GOOD_DOOR_REWARD if action == state \
                else self.BAD_DOOR_REWARD
            state = self.sample_start_state()

        else:  # not opening door
            obs = self.sample_observation(state, True)
            terminal = False
            reward = self.LISTEN_REWARD

        return POUCTInteraction(state, obs, reward, terminal)

    def step(self, action: int) -> EnvironmentInteraction:
        """ performs a step in the tiger environment given action

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             action: (`int`): 0 is open left, 1 is open right or 2 is listen

        RETURNS (`pobnrl.environments.EnvironmentInteraction`): the transition

        """

        transition = self.simulation_step(self.state, action)
        self._state = transition.state

        self._history.append({
            'action': action,
            'obs': transition.observation,
            'reward': transition.reward
        })

        return EnvironmentInteraction(
            transition.observation, transition.reward, transition.terminal
        )

    # pylint: disable=no-self-use
    def obs2index(self, observation: np.array) -> int:
        """ projects the observation as an int

        Args:
             observation: (`np.array`): observation to project

        RETURNS (`int`): int representation of observation

        """
        assert self._obs_space.contains(observation), \
            f"{observation} not in {self._obs_space}"

        assert np.sum(observation) in [0, 1], "wrong value for observation"

        if observation[0] == 1:
            return 0

        return 2 - observation[1]  # 2 if observation[1] is **also** 0

    @property
    def action_space(self) -> ActionSpace:
        """ a `pobnrl.environments.ActionSpace`([3]) space """
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `pobnrl.misc.DiscreteSpace`([1,1]) space """
        return self._obs_space

    def display_history(self):
        """ prints out transitions """

        def to_string(element: int) -> str:
            """ action, state or observation to string

            Args:
                 element: (`int`): action, string or observation

            RETURNS (`str`): string representation of element

            """
            return "L" if int(element) == self.LEFT else "R"

        descr = "Tiger (" + to_string(self._history[0]) + ") and heard: "
        for step in self._history[1:-1]:
            descr = descr + to_string(step["obs"][1])

        self.log(
            LogLevel.V2,
            f"{descr}, opened {to_string(self._history[-1]['action'])}"
        )
