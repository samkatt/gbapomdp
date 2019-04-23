""" chain domain environment """

import time
from typing import Dict

import numpy as np

from misc import DiscreteSpace, POBNRLogger, LogLevel

from .environment import Environment, EnvironmentInteraction
from .misc import ActionSpace


# pylint: disable=too-many-instance-attributes
class ChainDomain(Environment):
    """ the chain environment

    The environments are indexed by problem size N and actionimask W =
    Ber(0.5)^NxN,with S={0,1}^NxN and A={0,1}. he agent begins each episode
    inthe upper left-most state in the grid and deterministically falls one row
    per time step. The state encodes the agent’s row and column as a one-hot
    vectorst. The actions {0,1} move the agent left or right depending on the
    action mask W at state s, which remains fixed. The agent incurs a cost of
    0.01/N for moving right in all states except for the right-most, in which
    the reward is 1. The reward for action left is always zero. An episode ends
    after N time steps so that the optimal policy is to move right each step
    and receive a total return of 0.99; all other policies receive zero or
    negative return

    """

    logger = POBNRLogger(__name__)

    def __init__(self, size: int, verbose: bool):
        """ construct the chain environment

        Args:
             size: (`int`): size of the grid
             verbose: (`bool`): whether to be verbose or not (print to stdout)

        """

        assert size > 1, "Please enter domain size > 1"

        self._size = size
        self._verbose = verbose

        self._move_cost = .01 / self.size

        self._action_space = ActionSpace(2)
        self._observation_space = DiscreteSpace(
            [[2] * self.size for _ in range(self.size)]
        )

        # x, level (size-1...0)
        self._init_state = {'x': 0, 'y': self.size - 1}
        self._state = self._init_state.copy()

        self._action_mapping = np.random.binomial(1, .5, self.size)

        self._last_recording_time = 0
        self._recording = False
        self._history = []

    @property
    def state(self) -> Dict[str, int]:
        """ returns current state

        Args:

        RETURNS (`Dict[str, int]`): {'x', 'y'} positions of agent

        """
        return self._state

    @state.setter
    def state(self, state: Dict[str, int]):
        """ sets state

        Args:
             state: (`Dict[str, int]`):{'x', 'y'}

        """
        assert not self._recording
        assert self.size > state['x'] >= 0
        assert self.size > state['y'] > 0
        assert state['x'] <= (self.size - state['y'])

        self._state = state

    def sample_start_state(self) -> Dict[str, int]:
        """ samples the (deterministic) start state

        Args:

        RETURNS (`Dict[str, int]`):{'x', 'y'}

        """

        return self._init_state.copy()

    @property
    def size(self):
        """ returns size (length) of (square) grid """
        return self._size

    def state2observation(self, state: Dict[str, int] = None):
        """ returns a 1-hot encoding of the state

        Will use self.state if provided state is not given

        Args:
             state: (`Dict[int]`): the state {'x', 'y'}

        """

        # default value of state
        if state is None:
            state = self.state

        obs = np.zeros((self.size, self.size))
        obs[state['x'], state['y']] = 1

        return obs

    def reset(self):
        """ resets internal state and return first observation

        resets the intenral state randomly (0 or 1)
        returns [0,0] as a 'null' initial observation

        """

        self._state = self._init_state.copy()

        # if we were recording, output the history and stop
        if self._recording:
            self.display_history()
            self._recording = False

        # record episodes every so often
        if self._verbose and time.time() - self._last_recording_time > 5:
            self._last_recording_time = time.time()
            self._history = []
            self._recording = True

        return self.state2observation()

    def step(self, action: int) -> EnvironmentInteraction:
        """ performs a step in the domain depending on action

        Dependin gon the state, action can either move the agent left or rigth
        deterministically. The observation is a noise-less 1-hot-encoding of
        the state and the environment stops when the end is reached (on either
        x-axis)

        Args:
             action: (`int`): 0 is action A, 1 = action B

        RETURNS (`pobnrl.environments.environment.EnvironmentInteraction`): the transition

        """

        assert 2 >= action >= 0, "expecting action A or B"

        # move horizontally
        if action == self._action_mapping[self.state['x']]:
            self._state['x'] += 1
            reward = -self._move_cost
        else:
            self._state['x'] = max(0, self.state['x'] - 1)  # bound on grid
            reward = 0

        # move vertically
        self._state['y'] -= 1

        terminal = self.state['y'] == 0

        if terminal and self.state['x'] == self.size - 1:
            reward = 1

        if self._recording:
            self._history.append(self.state['x'])

        return EnvironmentInteraction(
            self.state, self.state2observation(), reward, terminal
        )

    def obs2index(self, observation: np.array) -> int:
        """ projects the observation as an int

        Args:
             observation: (`np.array`): observation to project

        RETURNS (`int`): int representation of observation

        """

        assert observation.shape == self.observation_space.shape, \
            f"expected {self.observation_space.shape}, got {observation.shape}"
        assert np.all(self.size > observation) and np.all(observation >= 0)
        assert np.sum(observation) == 1

        return observation.argmax()

    @property
    def action_space(self) -> DiscreteSpace:
        """ a `pobnrl.misc.DiscreteSpace` space with 2 actions"""
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `pobnrl.misc.DiscreteSpace` space of size x size """
        return self._observation_space

    def display_history(self):
        """ prints out transitions """

        descr = "0"

        for x in self._history:
            descr += f"->{x}"

        self.logger.log(LogLevel.V2, f"agent travelled {descr}")
