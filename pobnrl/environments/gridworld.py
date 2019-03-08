""" gridworld environment """

import copy
import time

import numpy as np
from environments.environment import Environment

from utils import math_space


class GridWorld(Environment):
    """ the tiger environment """

    # consts
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    CORRECT_OBSERVATION_PROB = .8
    MOVE_SUCCESS_PROB = .95
    SLOW_MOVE_SUCCESS_PROB = .15

    state = np.zeros(2)
    _slow_cells = set()

    ##
    # @brief hot-encoding of the actions
    action_to_vec = [[0, 1], [1, 0], [0, -1], [-1, 0]]

    # verbosity helpers
    action_to_string = ["^", ">", "v", "<"]

    # recording helpers
    _last_recording_time = 0
    _recording = False
    _history = []

    # FIXME: should accept specific arguments instead of conf
    def __init__(self, conf):

        assert conf.domain_size > 0
        assert conf.domain_size % 2 == 1

        # confs
        self._verbose = conf.verbose
        self._size = conf.domain_size

        self.goal_state = np.array([self._size - 1, self._size - 1])

        self._spaces = {
            "A": math_space.DiscreteSpace(
                [4]), "O": math_space.DiscreteSpace(
                    np.ones((self._size, self._size)).tolist())}

        # generate multinomial probabilities for the observation function (1-D)
        _obs_mult = [self.CORRECT_OBSERVATION_PROB]

        left_over_p = 1 - self.CORRECT_OBSERVATION_PROB
        for _ in range(int(self._size - 2)):
            left_over_p *= .5
            _obs_mult.append(left_over_p / 2)
            _obs_mult.insert(0, left_over_p / 2)

        _obs_mult.append(left_over_p / 2)
        _obs_mult.insert(0, left_over_p / 2)

        self._obs_mult = np.array(_obs_mult)

        # generate slow locations
        edge = self._size - 1

        # bottom left side for larger domains
        if self._size > 5:
            self._slow_cells.add((1, 1))

        if self._size == 3:
            self._slow_cells.add((1, 1))
        elif self._size < 7:
            self._slow_cells.add((edge - 1, edge - 2))
            self._slow_cells.add((edge - 2, edge - 1))
        else:
            self._slow_cells.add((edge - 1, edge - 3))
            self._slow_cells.add((edge - 3, edge - 1))
            self._slow_cells.add((edge - 2, edge - 2))

    def bound_in_grid(self, state_or_obs):
        """ makes sure input state or observation is bounded within <0,size> """
        return np.maximum(0, np.minimum(state_or_obs, self._size - 1))

    def generate_observation(self, state):
        """ samples an observation, an displacement from the current state """
        # [fixme] test gridworld.generate_observation

        # state + displacement (where displacement is centered through -
        # self._size)
        unbounded_obs = state + \
            (np.random.multinomial(1, self._obs_mult, size=2) == 1).argmax(1) - (self._size - 1)
        bounded_obs = self.bound_in_grid(unbounded_obs).astype(int)

        # 2-D hot encoding
        obs = np.zeros((self._size, self._size))
        obs[bounded_obs[0], bounded_obs[1]] = 1

        return obs

    def reset(self):
        """ resets state """

        self.state = np.zeros(2)

        # if we were recording, output the history and stop
        if self._recording:
            self.display_history()
            self._recording = False

        # record episodes every so often
        if self._verbose and time.time() - self._last_recording_time > 15:
            self._last_recording_time = time.time()
            self._history = [copy.deepcopy(self.state)]
            self._recording = True

        return self.generate_observation(self.state)

    def step(self, action):
        """ update state wrt action (listen or open) """

        if tuple(self.state) not in self._slow_cells:
            move_prob = self.MOVE_SUCCESS_PROB
        else:
            move_prob = self.SLOW_MOVE_SUCCESS_PROB

        if np.random.uniform() < move_prob:
            self.state = self.bound_in_grid(
                self.state + self.action_to_vec[int(action)])

        obs = self.generate_observation(self.state)
        terminal = (self.state == self.goal_state).all()
        reward = 1 if terminal else 0

        if self._recording:
            self._history.append({
                'action': action,
                'obs': obs,
                'reward': reward,
                'state': copy.deepcopy(self.state)})

        return obs, reward, terminal

    def spaces(self):
        """ A: 4, O: S """
        return self._spaces

    def obs_to_string(self, obs):
        """ translate hot-encoding of obs to string """
        return str(np.unravel_index(obs.argmax(), obs.shape))

    def display_history(self):
        """ prints out transitions """

        descr = "[0,0] "
        for step in self._history[1:]:
            descr += self.action_to_string[int(step["action"])] + " " + str(
                step["state"]) + " (" + self.obs_to_string(step["obs"]) + ") "

        print(descr)
