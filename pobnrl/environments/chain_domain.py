""" chain domain environment """

import logging
import time
from typing import List

import numpy as np

from environments.environment import Environment
from misc import DiscreteSpace, log_level


# pylint: disable=too-many-instance-attributes
class ChainDomain(Environment):
    """ the chain environment

    TODO: add doc

    """

    logger = logging.getLogger(__name__)

    def __init__(self, size: int, verbose: bool):
        """ construct the chain environment

        Args:
             size: (`int`): size of the grid
             verbose: (`bool`): whether to be verbose or not (print to stdout)

        """

        self._size = size
        self._verbose = verbose

        self._move_cost = .01 / self.size

        self._action_space = DiscreteSpace([2])
        self._observation_space = DiscreteSpace(
            [[2] * (self.size), [2] * (self.size)]
        )

        # x, level (size-1...0)
        self._init_state = [0, self.size - 1]
        self._state = self._init_state.copy()

        num_states = self.size * self.size
        self._move_effect = np.ones((num_states, 2))  # default is 'right'
        self._move_effect[
            np.arange(num_states), np.random.randint(0, 2, num_states)
        ] = -1  # randomly pick half to go left

        # [x,y,a] is effect of action a in (x,y)
        self._move_effect \
            = self._move_effect.reshape((self.size, self.size, 2)).astype(int)
        self._move_effect[0, self._move_effect[0] < 0] = 0  # bound in grid

        self._last_recording_time = 0
        self._recording = False
        self._history = []

    @property
    def state(self) -> List[int]:
        """ returns current state

        Args:

        RETURNS (`List[int]`): [0] = x of agent, [1] = level of env

        """
        return self._state

    @property
    def size(self):
        """ returns size (length) of (square) grid """
        return self._size

    def state2observation(self, state: List[int] = None):
        """ returns a 1-hot encoding of the state

        Will use self.state if provided state is not given

        Args:
             state: (`List[int]`): the state ([x, level])

        """

        # default value of state
        if state is None:
            state = self.state

        obs = np.zeros((2, self.size))

        obs[0, state[0]] = 1
        obs[1, state[1]] = 1

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
            self._recording = True

        return self.state2observation()

    def step(self, action: int) -> list:
        """ performs a step in the domain depending on action

        Dependin gon the state, action can either move the agent left or rigth
        deterministically. The observation is a noise-less 1-hot-encoding of
        the state and the environment stops when the end is reached (on either
        x-axis)

        Args:
             action: (`int`): 0 is action A, 1 action B.

        RETURNS (`list`): [observation, reward (float), terminal (bool)]

        """

        assert 2 >= action >= 0

        agent_x, agent_y = self._state

        self._state[0] += self._move_effect[agent_x, agent_y, action]
        self._state[1] -= 1

        agent_x, agent_y = self._state

        # either found the end or not
        if agent_x != self.size - 1:
            reward = - self._move_cost
            terminal = agent_y == 0
        else:
            assert agent_y == 0
            reward = 1
            terminal = True

        if self._recording:
            self._history.append(agent_x)

        return self.state2observation(), reward, terminal

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

        for state in self._history:
            descr += f"-> {state}"

        self.logger.log(log_level['verbose'], "agent travelled %s", descr)
