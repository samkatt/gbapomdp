""" collision avoidance environment """

import time

import copy
import numpy as np

from environments.environment import Environment
from misc import DiscreteSpace


# pylint: disable=too-many-instance-attributes
class CollisionAvoidance(Environment):
    """ the tiger environment """

    # const
    BLOCK_MOVE_PROB = .5
    COLLISION_REWARD = -1000

    action_to_move = [1, 0, -1]
    action_to_string = ["UP", "STAY", "DOWN"]

    def __init__(self, domain_size: int, verbose: bool):
        """ constructs a Collision Avoidance domain of specified size

        Args:
             domain_size: (`int`): the size of the grid
             verbose: (`bool`): whether to print experiences to stdout

        """

        assert domain_size > 0, "Domain size must be > 0"
        assert domain_size % 2 == 1

        self._verbose = verbose
        self._size = domain_size
        self._mid = int(self._size / 2)

        self.init_state = {
            'agent': np.array((self._size - 1, self._mid)),
            'obstacle': self._mid
        }

        self.state = copy.deepcopy(self.init_state)

        self._spaces = {
            "A": DiscreteSpace([3]),
            "O": DiscreteSpace([self._size])
        }

        self._last_recording_time = 0
        self._recording = False
        self._history = []

    @property
    def state(self):
        """ returns current state TODO: nyi """
        raise NotImplementedError

    def bound_in_grid(self, y_pos: int) -> int:
        """ returns bounded y_pos s.t. it is within the grid

        simpy returns y_pos if it is in the grid size,
        otherwise returns the edge value

        Args:
             y_pos: (`int`): some y position on the grid

        RETURNS (`int`): the bounded value of y

        """
        return max(0, min(self._size - 1, y_pos))

    def generate_observation(self, obstacle_pos: int) -> np.array:
        """ samples a possibly noisy observation of the obstacle

        Args:
             obstacle_pos: (`int`): a 'real' position of the obstacle

        RETURNS (`np.array`): an observation of the position [y]

        """
        obs = np.array(
            [self.bound_in_grid(round(obstacle_pos + np.random.normal()))])

        return obs

    def reset(self):
        """ resets state and potentially records the episode"""

        self.state = copy.deepcopy(self.init_state)

        # if we were recording, output the history and stop
        if self._recording:
            self.display_history()
            self._recording = False

        # record episodes every so often if verbose
        if self._verbose and time.time() - self._last_recording_time > 15:
            self._last_recording_time = time.time()
            self._history = [copy.deepcopy(self.state)]
            self._recording = True

        return self.generate_observation(self._mid)

    def step(self, action: int) -> list:
        """ updates the state and return observed transitions

        Will move the agent 1 cell to the left, and (depending on the action)
        up to 1 cell vertically. Stochastically move the obstacle,
        generating an observation in the process

        Is terminal when the agent reached the last column

        Args:
             action: (`int`): 0 is go down, 1 is stay or 2 is go up

        RETURNS (`list`): [observation, reward (float), terminal (bool)]

        """

        # move agent
        self.state['agent'][0] -= 1
        self.state['agent'][1] = self.bound_in_grid(
            self.state['agent'][1] + self.action_to_move[int(action)]
        )

        # move obstacle
        if np.random.random() < self.BLOCK_MOVE_PROB:
            if np.random.random() < .5:
                self.state['obstacle'] += 1
            else:
                self.state['obstacle'] -= 1

            self.state['obstacle'] = self.bound_in_grid(self.state['obstacle'])

        # observation
        obs = self.generate_observation(self.state['obstacle'])

        # reward and terminal
        reward = 0
        terminal = False

        if self.state['agent'][0] == 0:
            terminal = True
            if self.state['agent'][1] == self.state['obstacle']:
                reward = self.COLLISION_REWARD

        # recording
        if self._recording:
            self._history.append({
                'action': action,
                'obs': obs,
                'reward': reward,
                'state': copy.deepcopy(self.state)})

        return obs, reward, terminal

    def spaces(self) -> dict:
        """ spaces

        RETURNS (`dict`): {'O', 'A'} of discrete spaces |O| = size, |A| = 3

        """
        return self._spaces

    def display_history(self):
        """ prints out transitions """

        descr = f"[{self._size-1}, {self._mid}]"

        for step in self._history[1:]:
            descr += " " + \
                f"{self.action_to_string[int(step['action'])]} --> "\
                f"{step['state']['agent']} "\
                f"vs {step['state']['obstacle']} ({step['obs'][0]})"

        print(descr)
