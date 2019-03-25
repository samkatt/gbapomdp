""" tiger environment """

import time

import copy
import numpy as np

from environments.environment import Environment
from misc import DiscreteSpace


class Tiger(Environment):
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

        self._verbose = verbose
        self._state = self.sample_start_state()

        self._spaces = {
            "A": DiscreteSpace([3]),
            "O": DiscreteSpace([2, 2])
        }

        self._last_recording_time = 0
        self._recording = False
        self._history = []

    @property
    def state(self):
        """ returns current state """
        return self._state

    def sample_start_state(self) -> int:  # pylint: disable=no-self-use
        """ samples a random state (tiger left or right)

        RETURNS (`int`): an initial state (in [0,1])

        """
        return np.random.randint(0, 2)

    def sample_observation(self, listening: bool) -> np.array:
        """ samples an observation, listening stores whether agent is listening

        Args:
             listening: (`bool`): whether the agent is listening

        RETURNS (`np.array`): the observation (hot-encoded)

        """
        obs = np.zeros(self._spaces["O"].shape)

        # not listening means [0,0] observation (basically a 'null')
        if not listening:
            return obs

        # 1-hot-encoding
        if np.random.random() < self.CORRECT_OBSERVATION_PROB:
            obs[self.state] = 1
        else:
            obs[int(not self.state)] = 1

        return obs

    def reset(self):
        """ resets internal state and return first observation

        resets the intenral state randomly (0 or 1)
        returns [0,0] as a 'null' initial observation

        """

        self._state = self.sample_start_state()

        # if we were recording, output the history and stop
        if self._recording:
            self.display_history()
            self._recording = False

        # record episodes every so often
        if self._verbose and time.time() - self._last_recording_time > 5:
            self._last_recording_time = time.time()
            self._history = [copy.deepcopy(self.state)]
            self._recording = True

        return np.zeros(2)

    def step(self, action: int) -> list:
        """ performs a step in the tiger environment given action

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             action: (`int`): 0 is open left, 1 is open right or 2 is listen

        RETURNS (`list`): [observation, reward (float), terminal (bool)]

        """

        if action != self.LISTEN:
            obs = self.sample_observation(False)
            terminal = True
            reward = self.GOOD_DOOR_REWARD if action == self.state \
                else self.BAD_DOOR_REWARD

        else:  # not opening door
            obs = self.sample_observation(True)
            terminal = False
            reward = self.LISTEN_REWARD

        if self._recording:
            self._history.append(
                {'action': action, 'obs': obs, 'reward': reward})

        return obs, reward, terminal

    def spaces(self) -> dict:
        """ returns size of domain space {'O', 'A'}

        RETURNS (`dict`): {'O', 'A'} of spaces to sample from

        """
        return self._spaces

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

        print(descr + ", opened " + to_string(self._history[-1]["action"]))
