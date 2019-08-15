""" gridworld environment """

from collections import namedtuple
from typing import List, Tuple, Set, Optional
import random

import numpy as np

from environments import Environment, EnvironmentInteraction, ActionSpace
from environments import Simulator, SimulationResult, EncodeType
from misc import DiscreteSpace, POBNRLogger


class GridWorld(Environment, Simulator, POBNRLogger):
    """ the gridworld environment

    A 2-d grid world where the agent needs to go to a goal location (part of
    the state space). The agent has 4 actions, a step in each direction, that
    is carried out succesfully 95% of the time (and is a no-op otherwise),
    except for some 'bad' cells, where the successrate drops to 15%. The
    observation function is noisy, with gaussian probability around the agent's
    real location (that accumulates around the edges).

    """

    # consts
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    CORRECT_OBSERVATION_PROB = .8
    MOVE_SUCCESS_PROB = .95
    SLOW_MOVE_SUCCESS_PROB = .15

    action_to_vec = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    action_to_string = ["UP", "RIGHT", "DOWN", "LEFT"]

    class Goal(namedtuple('grid_goal', 'x y index')):
        """ goal in gridworld

            Contains:
                x: (`int`)
                y: (`int`)
                index: (`int`)
        """

        # required to keep lightweight implementation of namedtuple
        __slots__ = ()

    def __init__(
            self,
            domain_size: int,
            encoding: EncodeType,
            slow_cells: Optional[Set[Tuple[int, int]]] = None):
        """ creates a gridworld of provided size and verbosity

        Args:
             domain_size: (`int`): the size (assumed odd) of the grid
             encoding: (`EncodeType`): the observation encoding
             slow_cells: (`Set[Tuple[int, int]]`): the cells that are slow (default assignment if left None)

        """

        assert domain_size > 0

        POBNRLogger.__init__(self)

        # confs
        self._size = domain_size
        self._one_hot_goal_encoding = encoding == EncodeType.ONE_HOT

        # generate multinomial probabilities for the observation function (1-D)
        obs_mult = [self.CORRECT_OBSERVATION_PROB]

        left_over_p = 1 - self.CORRECT_OBSERVATION_PROB
        for _ in range(int(self.size - 2)):
            left_over_p *= .5
            obs_mult.append(left_over_p / 2)
            obs_mult.insert(0, left_over_p / 2)

        obs_mult.append(left_over_p / 2)
        obs_mult.insert(0, left_over_p / 2)

        self.obs_mult = np.array(obs_mult)

        # generate slow locations
        self._slow_cells \
            = slow_cells if slow_cells is not None else GridWorld.generate_slow_cells(self.size)

        import pdb; pdb.set_trace()

        # generate goal locations
        self._goal_cells: List[GridWorld.Goal] = []

        goal_edge_start = self.size - 2 if self.size < 5 \
            else self.size - 3 if self.size < 7 else self.size - 4

        edge = self.size - 1
        for pos in range(goal_edge_start, self.size - 1):
            self._goal_cells.append(GridWorld.Goal(  # fill top side
                pos,
                edge,
                len(self._goal_cells)
            ))

            self._goal_cells.append(GridWorld.Goal(  # fill right side
                edge,
                pos,
                len(self._goal_cells)
            ))

        self._goal_cells.append(  # top right corner
            GridWorld.Goal(edge, edge, len(self._goal_cells))
        )

        if self.size > 3:
            self._goal_cells.append(GridWorld.Goal(
                edge - 1,
                edge - 1,
                len(self._goal_cells)
            ))

        if self.size > 6:
            self._goal_cells.append(GridWorld.Goal(
                edge - 2,
                edge - 1,
                len(self._goal_cells)
            ))
            self._goal_cells.append(GridWorld.Goal(
                edge - 1,
                edge - 2,
                len(self._goal_cells)
            ))

        self._state_space = DiscreteSpace([self.size, self.size, len(self._goal_cells)])
        self._action_space = ActionSpace(4)

        if not self._one_hot_goal_encoding:
            self._obs_space = DiscreteSpace([self.size, self.size, len(self._goal_cells)])
        else:
            self._obs_space = DiscreteSpace(
                [self.size, self.size] + (2 * np.ones(len(self._goal_cells))).astype(int).tolist()
            )

        self._state = self.sample_start_state()

    @property
    def state(self) -> np.ndarray:
        """ returns current state

        Args:

        RETURNS (`np.ndarray`): [x,y,goal_index]

        """
        return self._state

    @state.setter
    def state(self, value: np.ndarray):
        """ set state of grid

        Checks whether the state is valid through assertion error

        Args:
             value: (`np.ndarray`): [x,y,goal_index]

        """

        agent_x, agent_y, goal_index = value

        assert len(self.goals) > goal_index >= 0
        assert 0 <= agent_x < self.size and 0 <= agent_x < self.size
        assert 0 <= agent_y < self.size and 0 <= agent_y < self.size

        self._state = np.array([agent_x, agent_y, goal_index], dtype=int)

    @property
    def state_space(self) -> DiscreteSpace:
        """ `pobnrl.misc.DiscreteSpace`([size,size,num_goals]) """
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        """ a `pobnrl.environments.ActionSpace`([4]) space """
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `pobnrl.misc.DiscreteSpace`([size,size] + ones * num_goals) """
        return self._obs_space

    @property
    def slow_cells(self) -> Set[Tuple[int, int]]:
        """ all the cells where the agent is slow on

        Args:

        RETURNS (`Set[Tuple[int, int]]`):

        """
        return self._slow_cells

    def sample_start_state(self) -> np.ndarray:
        """ returns [[0,0], some_goal]

        Args:

        RETURNS (`np.ndarray`): [x,y,goal_index]

        """
        return np.array([0, 0, self.sample_goal().index], dtype=int)

    @property
    def size(self) -> int:
        """ size of grid (square)

        RETURNS (`int`):

        """
        return self._size

    @property
    def goals(self) -> List['GridWorld.Goal']:
        """ returns the number of **possible** goals

        Args:

        RETURNS (`List[` `GridWorld.Goal` `]`):

        """
        return self._goal_cells

    def bound_in_grid(self, state_or_obs: np.ndarray) -> np.ndarray:
        """ returns bounded state or obs s.t. it is within the grid

        simpy returns state_or_obs if it is in the grid size,
        otherwise returns the edge value

        Args:
             state_or_obs: (`np.ndarray`): some (x,y) position on the grid

        RETURNS (`np.ndarray`): the bounded value state_or_obs

        """
        return np.maximum(0, np.minimum(state_or_obs, self.size - 1))

    def sample_goal(self) -> 'GridWorld.Goal':
        """ samples a goal position

        RETURNS (`Tuple[` `GridWorld.Goal` `]`): the goal state

        """
        return random.choice(self._goal_cells)

    def obs_noise(self) -> np.ndarray:
        """ returns the noise that comes with an observation

        RETURNS (`np.ndarray`): [x,y] int noise

        """

        return \
            np.random.multinomial(1, self.obs_mult, 2).argmax(axis=1) \
            - self.size + 1

    def generate_observation(self, state: np.ndarray) -> np.array:
        """ generates a noisy observation of the state

        Args:
             state: (`np.ndarray`): [x, y, goal_index]

        RETURNS (`np.array`): [x,y,... hot-encodign-goal-pos....]

        """

        # state + displacement, where displacement is centered through - size
        obs = self.bound_in_grid(state[:2] + self.obs_noise())

        if not self._one_hot_goal_encoding:
            return np.array([*obs, state[2]], dtype=int)

        # 1-hot-encoding goal
        goal_observation = np.zeros(len(self.goals), dtype=int)
        goal_observation[state[2]] = 1

        return np.hstack([obs, goal_observation])

    def reset(self) -> np.ndarray:
        """ resets state """

        self._state = self.sample_start_state()
        return self.generate_observation(self.state)

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """ performs a step on state and action and returns the transition

        moves agent accross the grid depending on the direction it took

        Args:
             state: (`np.ndarray`): [x, y, goal_index]
             action: (`int`): agent's taken action

        RETURNS (`pobnrl.environments.SimulationResult`): the transition

        """

        assert 4 > action >= 0, "agent can only move in 4 directions"

        agent_x, agent_y, goal_index = state
        agent_pos = np.array([agent_x, agent_y], dtype=int)

        if tuple(agent_pos) not in self.slow_cells:
            move_prob = self.MOVE_SUCCESS_PROB
        else:
            move_prob = self.SLOW_MOVE_SUCCESS_PROB

        if np.random.uniform() < move_prob:
            agent_pos = self.bound_in_grid(
                agent_pos + self.action_to_vec[int(action)])

        new_state = np.array([*agent_pos, goal_index], dtype=int)

        obs = self.generate_observation(new_state)

        return SimulationResult(new_state, obs)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """ 0, unless leaving the goal

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """

        # reward if terminal
        return int(self.terminal(state, action, new_state))

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """ False, unless leaving the goal

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`): whether the transition is terminal

        """

        assert self.state_space.contains(state)
        assert self.state_space.contains(new_state)
        assert self.action_space.contains(action)

        # terminal if [x,y] == [goal_x,goal_y]
        goal = self.goals[state[2]]
        return state[0] == goal.x and state[1] == goal.y

    def step(self, action: int) -> EnvironmentInteraction:
        """ update state as a result of action

        moves agent accross the grid depending on the direction it took

        Args:
             action: (`int`): agent's taken action

        RETURNS (`pobnrl.environments.EnvironmentInteraction`): the transition

        """

        sim_step = self.simulation_step(self.state, action)

        reward = self.reward(self.state, action, sim_step.state)
        terminal = self.terminal(self.state, action, sim_step.state)

        if self.log_is_on(POBNRLogger.LogLevel.V2):
            if self._one_hot_goal_encoding:
                goal_index = np.argmax(self.state[2:])
            else:
                goal_index = self.state[2]

            goal = self.goals[goal_index]

            self.log(
                POBNRLogger.LogLevel.V2,
                f"Agent moved from {self.state[:2]}  to {sim_step.state[:2]}"
                f" after picking {self.action_to_string[action]} and"
                f" observed {sim_step.observation[:2]}"
                f" (goal {goal})"
            )

        self._state = sim_step.state

        return EnvironmentInteraction(sim_step.observation, reward, terminal)

    def obs2index(self, observation: np.array) -> int:
        """ projects the observation as an int

        Args:
             observation: (`np.array`): observation to project

        RETURNS (`int`): int representation of observation

        """
        assert self.observation_space.contains(observation), \
            f"{observation} not in {self.observation_space}"
        assert \
            np.all(self.size > observation[:2]) \
            and len(self.goals) > observation[2] \
            and np.all(observation >= 0), \
            f"{observation} not in {self.observation_space}"

        if not self._one_hot_goal_encoding:
            return self._obs_space.index_of(observation)

        # one-hot goal encoding:
        assert np.sum(observation[2:]) == 1, "only 1 goal may be true"

        if observation[2] == 1:  # corner case: first potential goal
            return observation[0] + observation[1] * self.size

        # increment by goal
        return observation[0] \
            + observation[1] * self.size \
            + self.size * self.size * pow(2, np.argmax(observation[2:]) - 1)

    @staticmethod
    def generate_slow_cells(size: int) -> Set[Tuple[int, int]]:
        """ returns a set of cells that the agent are slow on, depending on `size`

        Args:
             size: (`int`): the size of the grid

        RETURNS (`Set[Tuple[int, int]]`):

        """

        slow_cells: Set[Tuple[int, int]] = set()

        edge = size - 1
        if size > 5:  # bottom left side for larger domains
            slow_cells.add((1, 1))

        if size == 3:
            slow_cells.add((1, 1))
        elif size < 7:
            slow_cells.add((edge - 1, edge - 2))
            slow_cells.add((edge - 2, edge - 1))
        else:
            slow_cells.add((edge - 1, edge - 3))
            slow_cells.add((edge - 3, edge - 1))
            slow_cells.add((edge - 2, edge - 2))

        return slow_cells
