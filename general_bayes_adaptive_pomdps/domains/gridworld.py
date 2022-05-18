"""Gridworld problem implemented as domain"""
import random
from logging import Logger
from typing import List, NamedTuple, Optional, Set, Tuple

import numpy as np

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace, LogLevel


class GridWorld(Domain):
    """The gridworld domain

    A 2-d grid world where the agent needs to go to a (variable) goal location.
    The agent has 4 actions, a step in each direction, that is carried out
    succesfully 95% of the time and is a no-op otherwise. However, there are
    (stationary) 'bad' cells in the grid, which reduce the successrate to 15%.

    The observation contains two quantities: the agent's and goal's location.
    The goal (index) is _always_ observed, with no uncertainty, however the
    agent's location is like a noisy GPS: there is some noise added which build
    up along the edges. The probability of correctly observing the correct x
    _or_ y position is set to 0.8, and the rest of the probability is spread,
    where the probability halves at each step.

    The domain allows for two 'forms' of representation depending on the
    construction input `one_hot_encode_goal`. If this is set to true, then the
    goal index/location is described (both in state and observation) in a
    one-hot encoding.
    """

    # consts
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    CORRECT_OBSERVATION_PROB = 0.8
    MOVE_SUCCESS_PROB = 0.95
    SLOW_MOVE_SUCCESS_PROB = 0.15

    action_to_x = [0, 1, 0, -1]
    action_to_y = [1, 0, -1, 0]
    action_to_string = ["UP", "RIGHT", "DOWN", "LEFT"]

    NOISE_SAMPLE_CACHE_SIZE = 5000

    class Goal(NamedTuple):
        """A goal in gridworld represented by its location and index"""

        x: int
        y: int
        index: int

    def __init__(
        self,
        domain_size: int,
        one_hot_encode_goal: bool,
        slow_cells: Optional[Set[Tuple[int, int]]] = None,
    ):
        """creates a gridworld of provided size and verbosity

        Args:
             domain_size: (`int`): the size (odd, positive) of the grid
             one_hot_encode_goal: (`bool`): the observation encoding
             slow_cells: (`Set[Tuple[int, int]]`): the cells that are slow (default assignment if left None)

        """
        assert domain_size > 0 and domain_size % 2 == 1

        super().__init__()
        self._logger = Logger(self.__class__.__name__)

        # confs
        self.size = domain_size
        self._one_hot_goal_encoding = one_hot_encode_goal

        # the noisey observation is fairly complex because
        #   (1) the distibution is non-trivial
        #   (2): we cache it for performance
        #  How the distribution is computed can be seen in :func:`generate_multinominal_noise`
        self._obs_mult = GridWorld.generate_multinominal_noise(
            GridWorld.CORRECT_OBSERVATION_PROB, self.size
        )

        # We use this to "move" the observation. The distribution is centered
        # around... the (list) center. However, we add noise by adding a `diff`
        # to the real observation, which means that the 'center' diff is zero.
        self._obs_diff = list(range(-self.size + 1, self.size))

        # This is a cache/database of 'noise' samples. Instead of computing the
        # noise each time, we commit to a relatively large compute
        # every-now-and-then. Here we store these (to be computed later).
        self._noise_samples = np.array([])

        # Will keep track of whether the cache must be updated: Whenever we
        # used the 'last' noise sample, we should re-compute the cache. We
        # start with the max value, since we leave the computation to the
        # actual noise computation function :func:`obs_noise`.
        self._noise_sample_index = GridWorld.NOISE_SAMPLE_CACHE_SIZE

        # generate slow locations
        self.slow_cells = (
            slow_cells
            if slow_cells is not None
            else GridWorld.generate_slow_cells(self.size)
        )

        # generate goal locations
        self.goals: List[GridWorld.Goal] = []

        goal_edge_start = (
            self.size - 2
            if self.size < 5
            else self.size - 3
            if self.size < 7
            else self.size - 4
        )

        edge = self.size - 1
        for pos in range(goal_edge_start, self.size - 1):
            self.goals.append(
                GridWorld.Goal(pos, edge, len(self.goals))  # fill top side
            )

            self.goals.append(
                GridWorld.Goal(edge, pos, len(self.goals))  # fill right side
            )

        self.goals.append(  # top right corner
            GridWorld.Goal(edge, edge, len(self.goals))
        )

        if self.size > 3:
            self.goals.append(GridWorld.Goal(edge - 1, edge - 1, len(self.goals)))

        if self.size > 6:
            self.goals.append(GridWorld.Goal(edge - 2, edge - 1, len(self.goals)))
            self.goals.append(GridWorld.Goal(edge - 1, edge - 2, len(self.goals)))

        self._state_space = DiscreteSpace([self.size, self.size, len(self.goals)])
        self._action_space = ActionSpace(4)

        if not self._one_hot_goal_encoding:
            self._obs_space = DiscreteSpace([self.size, self.size, len(self.goals)])
        else:
            self._obs_space = DiscreteSpace(
                [self.size, self.size] + [2] * len(self.goals)
            )

        self._state = self.sample_start_state()

    @property
    def obs_mult(self) -> np.ndarray:
        """The multinomial distribution noise of the observations"""
        return self._obs_mult

    @obs_mult.setter
    def obs_mult(self, v: np.ndarray):
        """Sets the multinomial noise distribution of the observations"""
        assert 0.99 < v.sum() < 1.001
        assert len(v) == len(self._obs_diff)

        # sets the 'cache index' to max:
        # at the first call for noise the cache will be re-computed
        # this will ensure that the new `v` will actually be used
        # as distribution, and not the old one due to cache
        self._noise_sample_index = GridWorld.NOISE_SAMPLE_CACHE_SIZE

        self._obs_mult = v

    @property
    def state(self) -> np.ndarray:
        """returns current state

        Args:

        RETURNS (`np.ndarray`): [x,y,goal_index]

        """
        return self._state

    @state.setter
    def state(self, value: np.ndarray):
        """set state of grid

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
        """:class:`general_bayes_adaptive_pomdps.misc.DiscreteSpace` ([size,size,num_goals])"""
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        """a :class:`general_bayes_adaptive_pomdps.core.ActionSpace` ([4]) space"""
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """a :class:`general_bayes_adaptive_pomdps.misc.DiscreteSpace` ([size,size] + ones * num_goals)"""
        return self._obs_space

    def sample_start_state(self) -> np.ndarray:
        """returns [[0,0], some_goal]

        Args:

        RETURNS (`np.ndarray`): [x,y,goal_index]

        """
        return np.array([0, 0, self.sample_goal().index], dtype=int)

    def bound_in_grid(self, x: int, y: int) -> Tuple[int, int]:
        """returns [x, y] bounded s.t. it is within the grid

        Retursn x / y [0, self.size - 1]

        Args:
            x: (`int`): some integer value (representing position x)
            y: (`int`): some integer value (representing position y)

        RETURNS (`x, y`): x / with minimum value 0 and maximum value size of grid

        """
        # very basic min/max, apparently quicker than either:
        #   (1): np.clip
        #   (2): min(max(lower_bound, x), higher_bound)
        x = 0 if x < 0 else self.size - 1 if x > self.size - 1 else x
        y = 0 if y < 0 else self.size - 1 if y > self.size - 1 else y

        return x, y

    def sample_goal(self) -> "GridWorld.Goal":
        """samples a goal position

        RETURNS (`Tuple[GridWorld.Goal]`): the goal state

        """
        return random.choice(self.goals)

    def obs_noise(self) -> np.ndarray:
        """returns the noise that comes with an observation

        RETURNS (`np.ndarray`): [x,y] int noise

        """
        if self._noise_sample_index == GridWorld.NOISE_SAMPLE_CACHE_SIZE:
            self._noise_samples = np.random.choice(
                self._obs_diff, (GridWorld.NOISE_SAMPLE_CACHE_SIZE, 2), p=self.obs_mult
            )
            self._noise_sample_index = 0

        self._noise_sample_index += 1
        return self._noise_samples[self._noise_sample_index - 1]

    def generate_observation(self, state: np.ndarray) -> np.ndarray:
        """generates a noisy observation of the state

        Args:
             state: (`np.ndarray`): [x, y, goal_index]

        RETURNS (`np.array`): [x,y,... hot-encodign-goal-pos....]

        """
        obs = np.zeros(self._obs_space.ndim, dtype=int)

        # state + displacement, where displacement is centered through - size
        obs[:2] = self.bound_in_grid(*state[:2] + self.obs_noise())

        if self._one_hot_goal_encoding:
            obs[state[2] + 2] = 1
        else:
            obs[2] = state[2]

        return obs

    def reset(self) -> np.ndarray:
        """resets state"""

        self._state = self.sample_start_state()
        return self.generate_observation(self.state)

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """performs a step on state and action and returns the transition

        moves agent accross the grid depending on the direction it took

        Args:
             state: (`np.ndarray`): [x, y, goal_index]
             action: (`int`): agent's taken action

        RETURNS (`general_bayes_adaptive_pomdps.core.SimulationResult`): the transition

        """
        assert 4 > action >= 0, "agent can only move in 4 directions"

        x, y, goal_index = state

        if (x, y) not in self.slow_cells:
            move_prob = self.MOVE_SUCCESS_PROB
        else:
            move_prob = self.SLOW_MOVE_SUCCESS_PROB

        if np.random.uniform() < move_prob:
            x, y = self.bound_in_grid(
                x + GridWorld.action_to_x[action], y + GridWorld.action_to_y[action]
            )

        new_state = np.array([x, y, goal_index], dtype=int)
        obs = self.generate_observation(new_state)

        return SimulationResult(new_state, obs)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """0, unless leaving the goal

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """

        # reward if terminal
        return int(self.terminal(state, action, new_state))

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """False, unless leaving the goal

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

    def step(self, action: int) -> DomainStepResult:
        """update state as a result of action

        moves agent accross the grid depending on the direction it took

        Args:
             action: (`int`): agent's taken action

        RETURNS (`general_bayes_adaptive_pomdps.core.EnvironmentInteraction`): the transition

        """

        sim_step = self.simulation_step(self.state, action)

        reward = self.reward(self.state, action, sim_step.state)
        terminal = self.terminal(self.state, action, sim_step.state)

        if self._logger.isEnabledFor(LogLevel.V2.value):
            if self._one_hot_goal_encoding:
                goal_index = np.argmax(self.state[2:])
            else:
                goal_index = self.state[2]

            goal = self.goals[goal_index]

            self._logger.log(
                LogLevel.V2.value,
                "Agent moved from %s to %s after picking %s and observed %s (goal %s)",
                self.state[:2],
                sim_step.state[:2],
                self.action_to_string[action],
                sim_step.observation[:2],
                goal,
            )

        self._state = sim_step.state

        return DomainStepResult(sim_step.observation, reward, terminal)

    @staticmethod
    def generate_slow_cells(size: int) -> Set[Tuple[int, int]]:
        """returns a set of cells that the agent are slow on, depending on ``size``

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

    @staticmethod
    def generate_multinominal_noise(correct_prob: float, grid_size: int) -> np.ndarray:
        """Generate a (1-D) multinominal distribution

        This distribution is used to sample _noise_, or how much an observation
        is 'off' (shifted). It puts ``correct_prob`` in the 'middle' of the
        distribution, and exponentially decreasing probabilities towards the
        start/end.

        Args:
            correct_prob: (`float`): probability of _correct_ observation [0,1]
            grid_size: (`int`): size of grid

        RETURNS (`np.ndarray`): (1 + 2 * grid_size,) probabilities
        """
        assert 0 < correct_prob < 1
        assert grid_size > 2

        ret = [correct_prob]
        left_over_prob = 1 - correct_prob

        for _ in range(grid_size - 2):
            left_over_prob /= 2
            ret.append(left_over_prob / 2)
            ret.insert(0, left_over_prob / 2)

        ret.append(left_over_prob / 2)
        ret.insert(0, left_over_prob / 2)

        return np.array(ret)


class GridWorldPrior(DomainPrior):
    """a prior that returns gridworlds without slow cells

    The slow cells are sampled with 1/3 chance, meaning that each location has
    a .333 chance of being a slow cell

    """

    def __init__(self, size: int, one_hot_encode_goal: bool):
        """a prior for :class:`general_bayes_adaptive_pomdps.domains.gridworld.GridWorld` of ``size`` with ``encoding``

        Args:
             size: (`int`):
             one_hot_encode_goal: (`bool`):

        """
        super().__init__()

        self._grid_size = size
        self._one_hot_encode_goal = one_hot_encode_goal

    def sample(self) -> Domain:
        """samples a :class:`general_bayes_adaptive_pomdps.domains.gridworld.GridWorld`

        Gridworld is of given size and encoding with a random set of slow cells
        """

        slow_cells: Set[Tuple[int, int]] = set()

        for i in range(self._grid_size):
            for j in range(self._grid_size):
                if random.random() < 1 / 3:
                    slow_cells.add((i, j))

        return GridWorld(self._grid_size, self._one_hot_encode_goal, slow_cells)
