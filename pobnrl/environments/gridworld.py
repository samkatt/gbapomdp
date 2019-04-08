""" gridworld environment """

import copy
import logging
import random
import time

from typing import List
import numpy as np

from environments.environment import Environment
from misc import DiscreteSpace, ActionSpace, log_level


class GridWorld(Environment):  # pylint: disable=too-many-instance-attributes
    """ the gridworld environment

    A 2-d grid world where the agent needs to go to a goal location (part of
    the state space). The agent has 4 actions, a step in each direction, that
    is carried out succesfully 95% of the time (and is a no-op otherwise),
    except for some 'bad' cells, where the successrate drops to 15%. The
    observation function is noisy, with some sort of gaussian probability
    around the agent's real location (that accumulates around the edges).

    """

    logger = logging.getLogger(__name__)

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
                [self._size, self._size]
                + (2 * np.ones(len(self._goal_cells))).tolist()
            )
        }

        self._state = [np.zeros(2), self.sample_goal()]

    @property
    def state(self):
        """ returns current state """
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

        self._state = [agent_pos, goal_pos]

    @property
    def size(self) -> int:
        """ size of grid (square)

        RETURNS (`int`):

        """
        return self._size

    @property
    def goals(self) -> List[int]:
        """ returns the number of **possible** goals

        Args:

        RETURNS (`List[int]`):

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

    def sample_goal(self) -> tuple:
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

        self._state = [np.zeros(2), self.sample_goal()]

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

    def step(self, action: int) -> list:
        """ update state wrt action """

        agent_pos, goal_pos = self.state

        # episode terminates when we have arrived in the goal state previously
        terminal = (agent_pos == goal_pos).all()

        if tuple(agent_pos) not in self._slow_cells:
            move_prob = self.MOVE_SUCCESS_PROB
        else:
            move_prob = self.SLOW_MOVE_SUCCESS_PROB

        if np.random.uniform() < move_prob:
            agent_pos = self.bound_in_grid(
                agent_pos + self.action_to_vec[int(action)])

        self._state = [agent_pos, goal_pos]

        obs = self.generate_observation(*self.state)
        reward = 1 if terminal else 0

        if self._recording:
            self._history.append({
                'action': action,
                'obs': obs,
                'reward': reward,
                'state': copy.deepcopy(self.state)})

        return obs, reward, terminal

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

        self.logger.log(log_level['verbose'], descr)
