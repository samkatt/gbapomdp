""" gridworld environment """

import copy
import random
import time

from typing import List, Tuple
import numpy as np

from misc import DiscreteSpace, POBNRLogger, LogLevel

from .misc import ActionSpace
from .environment import Environment, EnvironmentInteraction
from .environment import Simulator, SimulatedInteraction


class GridWorld(Environment, Simulator):  # pylint: disable=too-many-instance-attributes
    """ the gridworld environment

    A 2-d grid world where the agent needs to go to a goal location (part of
    the state space). The agent has 4 actions, a step in each direction, that
    is carried out succesfully 95% of the time (and is a no-op otherwise),
    except for some 'bad' cells, where the successrate drops to 15%. The
    observation function is noisy, with some sort of gaussian probability
    around the agent's real location (that accumulates around the edges).

    """

    logger = POBNRLogger(__name__)

    # consts
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    CORRECT_OBSERVATION_PROB = .8
    MOVE_SUCCESS_PROB = .95
    SLOW_MOVE_SUCCESS_PROB = .15

    action_to_vec = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    action_to_string = ["^^^", ">>>", "vvv", "<<<"]

    def __init__(self, domain_size: int, verbose: bool):
        """ creates a gridworld of provided size and verbosity

        Args:
             domain_size: (`int`): the size (assumed odd) of the grid
             verbose: (`bool`): whether to be verbose

        """

        assert domain_size > 0

        # confs
        self._verbose = verbose
        self._size = domain_size

        self._last_recording_time = 0
        self._recording = False
        self._history = []

        # generate multinomial probabilities for the observation function (1-D)
        obs_mult = [self.CORRECT_OBSERVATION_PROB]

        left_over_p = 1 - self.CORRECT_OBSERVATION_PROB
        for _ in range(int(self._size - 2)):
            left_over_p *= .5
            obs_mult.append(left_over_p / 2)
            obs_mult.insert(0, left_over_p / 2)

        obs_mult.append(left_over_p / 2)
        obs_mult.insert(0, left_over_p / 2)

        self.obs_mult = np.array(obs_mult)

        # generate slow locations
        self._slow_cells = set()

        edge = self._size - 1

        if self._size > 5:  # bottom left side for larger domains
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

        # generate goal locations
        self._goal_cells = []

        goal_edge_start = self._size - 2 if self._size < 5 \
            else self._size - 3 if self._size < 7 else self._size - 4

        for pos in range(goal_edge_start, self._size - 1):
            self._goal_cells.append((pos, edge))  # fill top side
            self._goal_cells.append((edge, pos))  # fill right side

        self._goal_cells.append((edge, edge))  # top right corner

        if self._size > 3:
            self._goal_cells.append((edge - 1, edge - 1))

        if self._size > 6:
            self._goal_cells.append((edge - 2, edge - 1))
            self._goal_cells.append((edge - 1, edge - 2))

        self._spaces = {
            "A": ActionSpace(4),
            "O": DiscreteSpace(
                [self._size, self._size] + (2 * np.ones(len(self._goal_cells))).tolist()
            )
        }

        self._state = [np.zeros(2), self.sample_goal()]

    @property
    def state(self) -> list:
        """ returns current state

        Args:

        RETURNS (`list`):

        """
        return self._state

    @state.setter
    def state(self, value: list):
        """ set state of grid

        Checks whether the state is valid through assertion error

        Args:
             value: (`list`): [np.array, (goal_x, goal_y)]

        """

        agent_pos, goal_pos = value
        assert goal_pos in self.goals
        assert agent_pos.shape == (2,)
        assert 0 <= agent_pos[0] < self.size and 0 <= agent_pos[1] < self.size
        assert not self._recording

        self._state = [agent_pos, goal_pos]

    def sample_start_state(self) -> list:
        """ returns [[0,0], some_goal]

        Args:

        RETURNS (`list`): [np.array, (goal_x, goal_y)]

        """
        return [np.zeros(2), self.sample_goal()]

    @property
    def size(self) -> int:
        """ size of grid (square)

        RETURNS (`int`):

        """
        return self._size

    @property
    def goals(self) -> List[Tuple[int]]:
        """ returns the number of **possible** goals

        Args:

        RETURNS (`List[Tuple[int]]`): list of (x,y)

        """
        return self._goal_cells

    def bound_in_grid(self, state_or_obs: np.array) -> np.array:
        """ returns bounded state or obs s.t. it is within the grid

        simpy returns state_or_obs if it is in the grid size,
        otherwise returns the edge value

        Args:
             state_or_obs: (`np.array`): some (x,y) position on the grid

        RETURNS (`np.array`): the bounded value state_or_obs

        """
        return np.maximum(0, np.minimum(state_or_obs, self._size - 1))

    def sample_goal(self) -> Tuple[int]:
        """ samples a goal position

        RETURNS (`Tuple[int]`): the goal state (x,y)

        """
        return random.choice(self._goal_cells)

    def generate_observation(
            self,
            agent_pos: np.array,
            goal_pos: tuple) -> np.array:
        """ generates a noisy observation of the state

        Args:
             agent_pos: (`np.array`): [x,y] position of the agent
             goal_pos: (`Tuple[int]`): (x,y) position of the goal

        RETURNS (`np.array`): [x,y,... hot-encodign-goal-pos....]

        """
        # state + displacement
        # where displacement is centered through - size
        unbounded_obs = agent_pos \
            + np.random.choice(len(self.obs_mult), p=self.obs_mult, size=2) \
            - (self._size - 1)

        bounded_obs = self.bound_in_grid(unbounded_obs).astype(int)

        # 1-hot-encoding goal
        goal_observation = np.zeros(len(self._goal_cells))
        goal_observation[self._goal_cells.index(goal_pos)] = 1

        return np.hstack([bounded_obs, goal_observation])

    def reset(self):
        """ resets state """

        self._state = self.sample_start_state()

        # if we were recording, output the history and stop
        if self._recording:
            self.display_history()
            self._recording = False

        # record episodes every so often
        if self._verbose and time.time() - self._last_recording_time > 15:
            self._last_recording_time = time.time()
            self._history = [copy.deepcopy(self.state)]
            self._recording = True

        return self.generate_observation(*self.state)

    def simulation_step(self, state: list, action: int) -> SimulatedInteraction:
        """ performs a step on state and action and returns the transition

        moves agent accross the grid depending on the direction it took

        Args:
             state: (`list`): [np.array, (goal_x, goal_y)]
             action: (`int`): agent's taken action

        RETURNS (`pobnrl.environments.environment.SimulatedInteraction`): the transition

        """

        assert 4 > action >= 0, "agent can only move in 4 directions"

        agent_pos, goal_pos = state

        # episode terminates when we have arrived in the goal state previously
        terminal = np.all(agent_pos == goal_pos)

        if tuple(agent_pos) not in self._slow_cells:
            move_prob = self.MOVE_SUCCESS_PROB
        else:
            move_prob = self.SLOW_MOVE_SUCCESS_PROB

        if np.random.uniform() < move_prob:
            agent_pos = self.bound_in_grid(
                agent_pos + self.action_to_vec[int(action)])

        state = [agent_pos, goal_pos]

        obs = self.generate_observation(*state)
        reward = 1 if terminal else 0

        return SimulatedInteraction(
            state, obs, reward, terminal
        )

    def step(self, action: int) -> EnvironmentInteraction:
        """ update state as a result of action

        moves agent accross the grid depending on the direction it took

        Args:
             action: (`int`): agent's taken action

        RETURNS (`pobnrl.environments.environment.EnvironmentInteraction`): the transition

        """

        transition = self.simulation_step(self.state, action)
        self._state = transition.state

        if self._recording:
            self._history.append({
                'action': action,
                'obs': transition.observation,
                'reward': transition.reward,
                'state': copy.deepcopy(self.state)})

        return EnvironmentInteraction(
            transition.observation, transition.reward, transition.terminal
        )

    def obs2index(self, observation: np.array) -> int:
        """ projects the observation as an int

        Args:
             observation: (`np.array`): observation to project

        RETURNS (`int`): int representation of observation

        """
        assert observation.shape == self.observation_space.shape, \
            f"expected {self.observation_space.shape}, got {observation.shape}"
        assert np.sum(observation[2:]) == 1, "only 1 goal may be true"
        assert np.all(self.size > observation) and np.all(observation >= 0)

        if observation[2] == 1:
            return observation[0] + observation[1] * self.size

        # increment by goal
        return observation[0] \
            + observation[1] * self.size \
            + self.size * self.size * pow(2, np.argmax(observation[2:]) - 1)

    @property
    def action_space(self) -> DiscreteSpace:
        """ a `pobnrl.misc.DiscreteSpace`([4]) space """
        return self._spaces['A']

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `pobnrl.misc.DiscreteSpace`([size,size] + ones * num_goals) """
        return self._spaces['O']

    def display_history(self):
        """ prints out transitions """

        descr = f"with goal {self._history[0][1]}:\n [0,0]"
        for step in self._history[1:]:
            descr += f" and {self.action_to_string[int(step['action'])]} " \
                + f"to:\n{step['state'][0]}({step['obs'][:2]})"

        self.logger.log(LogLevel.V2, descr)
