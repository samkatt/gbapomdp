""" tiger environment """

import numpy as np
from environments.environment import Environment

from utils import math_space

class Tiger(Environment):
    """ the tiger environment """

    # consts
    LISTEN = 2

    GOOD_DOOR_REWARD = 10
    BAD_DOOR_REWARD = -100

    LISTEN_REWARD = -0.1
    CORRECT_OBSERVATION_PROB = .85

    def __init__(self):
        self.state = self.sample_start_state()

        self._spaces = {
            "A": math_space.DiscreteSpace([3]),
            "O": math_space.DiscreteSpace([2])
        }

    # pylint: disable=R0201
    def sample_start_state(self):
        """ returns a random state (tiger left or right) """
        return np.random.randint(0,2)

    def sample_observation(self, listening):
        """ samples an observation, listening is true if agent is performing that action """
        if not listening:
            return self._spaces["O"].sample()

        if np.random.random() < self.CORRECT_OBSERVATION_PROB:
            return self.state
        else:
            return int(not self.state)

    def reset(self):
        """ resets state """
        self.state = self.sample_start_state()

        return self._spaces["O"].sample()

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

        return obs, reward, terminal

    def spaces(self):
        """ A: 3, O: 2 """
        return self._spaces
