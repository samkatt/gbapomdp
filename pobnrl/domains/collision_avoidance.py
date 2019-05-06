""" collision avoidance environment """

import time

from typing import List, Any
import copy
import numpy as np

from environments import Environment, EnvironmentInteraction, ActionSpace
from environments import POUCTSimulator, POUCTInteraction
from misc import DiscreteSpace, POBNRLogger, LogLevel


# pylint: disable=too-many-instance-attributes
class CollisionAvoidance(Environment, POUCTSimulator, POBNRLogger):
    """ the collision avoidance environment


    the agent pilots a plane that flies from right to left (one cell at a time)
    in a square grid. The agent can choose to stay level for no cost, or move
    either one cell diagonally with a reward of −1. The episode ends when the
    plane reaches the last column, where it must avoid collision with a
    vertically moving obstacle (or face a reward of −1000). The obstacle
    movement is stochastic, and the agent observes its own coordinates precisly
    and the obstacles coordinate with some noise.

    """

    # const
    BLOCK_MOVE_PROB = .5
    COLLISION_REWARD = -1000

    action_to_move = [-1, 0, 1]
    action_to_string = ["DOWN", "STAY", "UP"]

    def __init__(self, domain_size: int, verbose: bool):
        """ constructs a Collision Avoidance domain of specified size

        Args:
             domain_size: (`int`): the size of the grid
             verbose: (`bool`): whether to print experiences to stdout

        """

        assert domain_size > 0, "Domain size must be > 0"
        assert domain_size % 2 == 1, "Domain size must be odd"

        POBNRLogger.__init__(self)

        self._verbose = verbose
        self._size = domain_size
        self._mid = int(self._size / 2)

        self.init_state = {
            'agent_x': self._size - 1,
            'agent_y': self._mid,
            'obstacle': self._mid
        }

        self._state = copy.deepcopy(self.init_state)

        self._action_space = ActionSpace(3)
        self._obs_space = DiscreteSpace([self._size, self._size, self._size])

        self._last_recording_time = 0
        self._recording = False
        self._history: List[Any] = []

    @property
    def size(self):
        """ returns the size (of grid) of collision avoidance """
        return self._size

    @property
    def state(self) -> dict:
        """ {'agent_x': `int`, 'agent_y': `int`, 'obstacle': `int` } """
        return self._state

    @state.setter
    def state(self, state: dict):
        """ sets state

        Args:
             dict: {'agent_x': `int`, 'agent_y': `int`, 'obstacle': `int` }

        """

        assert self._size > state['agent_x'] > 0
        assert self._size > state['agent_y'] >= 0
        assert self._size > state['obstacle'] >= 0
        assert not self._recording

        self._state = state

    def sample_start_state(self) -> dict:
        """ returns the (deterministic) start state

        Args:

        RETURNS (`dict`): {'agent_x': `int`, 'agent_y': `int`, 'obstacle': `int` }

        """
        return copy.deepcopy(self.init_state)

    def bound_in_grid(self, y_pos: int) -> int:
        """ returns bounded y_pos s.t. it is within the grid

        simpy returns y_pos if it is in the grid size,
        otherwise returns the edge value

        Args:
             y_pos: (`int`): some y position on the grid

        RETURNS (`int`): the bounded value of y

        """
        return max(0, min(self._size - 1, y_pos))

    def generate_observation(self, state: dict = None) -> np.ndarray:
        """ generates an observation of the state (noisy obstacle sensor)

        Args:
             state: (`dict`): {'agent_x': int, 'agent_y': int, 'obstacle': int}
                                If None, it will use the current state

        RETURNS (`np.ndarray`): [agent_x, agent_y, obstacle_y]

        """
        if state is None:
            state = self.state

        obs = self.bound_in_grid(round(state['obstacle'] + np.random.normal()))

        return np.array([state['agent_x'], state['agent_y'], obs])

    def reset(self):
        """ resets state and potentially records the episode"""

        self._state = copy.deepcopy(self.init_state)

        # if we were recording, output the history and stop
        if self._recording:
            self.display_history()
            self._recording = False

        # record episodes every so often if verbose
        if self._verbose and time.time() - self._last_recording_time > 15:
            self._last_recording_time = time.time()
            self._history = [copy.deepcopy(self.state)]
            self._recording = True

        return self.generate_observation()

    def simulation_step(self, state: dict, action: int) -> POUCTInteraction:
        """ simulates stepping from state using action. Returns interaction

        Args:
             state: (`dict`): {'agent_x': int, 'agent_y': int, 'obstacle': int}
             action: (`int`): 0 is go down, 1 is stay or 2 is go up

        RETURNS (`pobnrl.environments.POUCTInteraction`):

        """
        assert 0 <= action < 3

        # move agent
        state['agent_x'] -= 1
        state['agent_y'] = self.bound_in_grid(
            state['agent_y'] + self.action_to_move[int(action)]
        )

        # move obstacle
        if np.random.random() < self.BLOCK_MOVE_PROB:
            if np.random.random() < .5:
                state['obstacle'] += 1
            else:
                state['obstacle'] -= 1

            state['obstacle'] = self.bound_in_grid(state['obstacle'])

        # observation
        obs = self.generate_observation(state)

        # reward and terminal
        reward = 0 if action == 1 else -1
        terminal = False

        if state['agent_x'] == 0:
            terminal = True
            if state['agent_y'] == state['obstacle']:
                reward = self.COLLISION_REWARD

        return POUCTInteraction(state, obs, reward, terminal)

    def step(self, action: int) -> EnvironmentInteraction:
        """ updates the state and return observed transitions

        Will move the agent 1 cell to the left, and (depending on the action)
        up to 1 cell vertically. Stochastically move the obstacle,
        generating an observation in the process

        Is terminal when the agent reached the last column

        Args:
             action: (`int`): 0 is go down, 1 is stay or 2 is go up

        RETURNS (`pobnrl.environments.EnvironmentInteraction`): the transition

        """
        assert 0 <= action < 3

        transition = self.simulation_step(self.state, action)
        self._state = transition.state

        # recording
        if self._recording:
            self._history.append({
                'action': action,
                'obs': transition.observation,
                'reward': transition.reward,
                'state': copy.deepcopy(transition.state)})

        return EnvironmentInteraction(
            transition.observation, transition.reward, transition.terminal
        )

    def obs2index(self, observation: np.array) -> int:
        """ projects the observation as an int

        Args:
             observation: (`np.array`): observation to project

        RETURNS (`int`): int representation of observation

        """
        assert self.observation_space.contains(observation), \
            f"{observation} not in space {self.observation_space}"
        assert np.all(self.size > observation) and np.all(observation >= 0), \
            f"expecting all observation to be more than 0, {observation}"

        return observation[0] \
            + observation[1] * self.size \
            + observation[2] * self.size * self.size

    @property
    def action_space(self) -> ActionSpace:
        """ a `pobnrl.environments.ActionSpace`([3]) space """
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `pobnrl.misc.DiscreteSpace`([grid_height]) space """
        return self._obs_space

    def display_history(self):
        """ prints out transitions """

        descr = f"[{self._size-1}, {self._mid}]"

        for step in self._history[1:]:
            descr += " " + \
                f"{self.action_to_string[int(step['action'])]} --> "\
                f"({step['state']['agent_x']}, {step['state']['agent_y']} "\
                f"({step['state']['obstacle']}: {step['obs'][0]})"

        self.log(LogLevel.V2, descr)
