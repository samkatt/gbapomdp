""" tiger environment """

import numpy as np
from environments.environment import Environment

from utils import math_space

import time

class Tiger(Environment):
    """ the tiger environment """

    # consts
    LEFT = 0
    LISTEN = 2

    GOOD_DOOR_REWARD = 10
    BAD_DOOR_REWARD = -100

    LISTEN_REWARD = -0.1
    CORRECT_OBSERVATION_PROB = .85

    def __init__(self, conf):

        # verbosity settings
        self._last_recording_time = 0
        self._verbose = conf.verbose
        self._recording = conf.verbose

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

        # verbosity handling
        if self._verbose and time.time() - self._last_recording_time > 15:
            self._last_recording_time = time.time()
            self._recording = True
        else:
            self._recording = False

        self.state = self.sample_start_state()

        return np.array([self._spaces["O"].sample()])

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
            self.record_transition(self.state, action, obs, reward, terminal)

        return np.array(obs), reward, terminal

    def spaces(self):
        """ A: 3, O: 2 """
        return self._spaces

    def record_transition(
            self,
            state,
            action,
            obs,
            reward,
            terminal):
        """ prints out transitions """

        state = "left" if state == self.LEFT else "right"
        obs = "left" if obs == self.LEFT else "right"

        if action == self.LISTEN:
            action = "listens"
        else:
            action = "opens left" if action == self.LEFT else "opens right"

        print("tiger " + state + ": agent " + action + " and hears " + obs)
