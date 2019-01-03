""" tiger environment """

import copy
import time

import numpy as np
from environments.environment import Environment

from utils import math_space

class Tiger(Environment):
    """ the tiger environment """

    # consts
    LEFT = 0
    LISTEN = 2

    GOOD_DOOR_REWARD = 10
    BAD_DOOR_REWARD = -100

    LISTEN_REWARD = -0.1
    CORRECT_OBSERVATION_PROB = .85

    _last_recording_time = 0
    _recording = False
    _history = []

    def __init__(self, conf):

        # verbosity settings
        self._verbose = conf.verbose

        self.state = self.sample_start_state()

        self._spaces = {
            "A": math_space.DiscreteSpace([3]),
            "O": math_space.DiscreteSpace([1, 1])
        }

    # pylint: disable=R0201
    def sample_start_state(self):
        """ returns a random state (tiger left or right) """
        return np.random.randint(0,2)

    def sample_observation(self, listening):
        """ samples an observation, listening is true if agent is performing that action """
        obs = np.zeros(self._spaces["O"].shape)

        if not listening:
            return obs

        if np.random.random() < self.CORRECT_OBSERVATION_PROB:
            obs[self.state] = 1
        else:
            obs[int(not self.state)] = 1

        return obs

    def reset(self):
        """ resets state """

        self.state = self.sample_start_state()

        # if we were recording, output the history and stop
        if self._recording:
            self.display_history()
            self._recording = False

        # record episodes every so often
        if self._verbose and time.time() - self._last_recording_time > 15:
            self._last_recording_time = time.time()
            self._history = [copy.deepcopy(self.state)]
            self._recording = True

        return np.zeros(2)

    def step(self, action):
        """ update state wrt action (listen or open) """

        if action != self.LISTEN:
            obs = self.sample_observation(False)
            terminal = True
            reward = self.GOOD_DOOR_REWARD if action == self.state else self.BAD_DOOR_REWARD

        else: # not opening door
            obs = self.sample_observation(True)
            terminal = False
            reward = self.LISTEN_REWARD

        if self._recording:
            self._history.append({'action': action, 'obs': obs, 'reward': reward})

        return obs, reward, terminal

    def spaces(self):
        """ A: 3, O: 2 """
        return self._spaces

    def display_history(self):
        """ prints out transitions """

        def to_string(element):
            """ return string rep of the element (action, state, or obs element) """
            return "L" if int(element) == self.LEFT else "R"

        descr = "Tiger (" + to_string(self._history[0]) + ") and heard: "
        for step in self._history[1:-1]:
            descr = descr + to_string(step["obs"][1])

        print(descr + ", opened " + to_string(self._history[-1]["action"]))
