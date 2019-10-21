""" collision avoidance environment """

from typing import Tuple, Optional
import numpy as np

from po_nrl.environments import Environment, EnvironmentInteraction, ActionSpace
from po_nrl.environments import Simulator, SimulationResult
from po_nrl.misc import DiscreteSpace, POBNRLogger


class CollisionAvoidance(Environment, Simulator, POBNRLogger):
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

    def __init__(self, domain_size: int, obstacle_policy: Optional[Tuple[float, float, float]] = None):
        """ constructs a Collision Avoidance domain of specified size

        Args:
             domain_size: (`int`): the size of the grid

        """

        assert domain_size > 0, "Domain size must be > 0"
        assert domain_size % 2 == 1, "Domain size must be odd"

        POBNRLogger.__init__(self)

        self._size = domain_size
        self._mid = int(self._size / 2)
        self._block_policy = obstacle_policy if obstacle_policy else (.25, .5, .25)

        assert abs(np.sum(self._block_policy) - 1) < .0001, \
            f"block policy {self._block_policy} must be a proper distribution"

        self._state_space = DiscreteSpace([self._size, self._size, self._size])
        self._action_space = ActionSpace(3)
        self._obs_space = self._state_space

        self._state = self.sample_start_state()

    @property
    def size(self):
        """ returns the size (of grid) of collision avoidance """
        return self._size

    @property
    def state(self) -> np.ndarray:
        """ [x, y, obs_y] """
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """ sets state

        Args:
             dict: [x, y, obs_y]

        """

        assert np.all(self._size > state)
        assert np.all(state >= 0)

        self._state = state

    @property
    def state_space(self) -> DiscreteSpace:
        """ `po_nrl.misc.DiscreteSpace`([x, y, obs_y]) """
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        """ a `po_nrl.environments.ActionSpace`([3]) space """
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `po_nrl.misc.DiscreteSpace`([size,size,size]) space """
        return self._obs_space

    def sample_start_state(self) -> np.ndarray:
        """ returns the (deterministic) start state

        Args:

        RETURNS (`np.ndarray`): [x, y, obs_y]

        """
        return np.array([self._size - 1, self._mid, self._mid])

    def bound_in_grid(self, y_pos: int) -> int:
        """ returns bounded y_pos s.t. it is within the grid

        simpy returns y_pos if it is in the grid size,
        otherwise returns the edge value

        Args:
             y_pos: (`int`): some y position on the grid

        RETURNS (`int`): the bounded value of y

        """
        return max(0, min(self._size - 1, y_pos))

    def generate_observation(self, state: np.ndarray = None) -> np.ndarray:
        """ generates an observation of the state (noisy obstacle sensor)

        Args:
             state: (`np.ndarray`): [x, y, obs_y] (if None, it will use the current state)

        RETURNS (`np.ndarray`): [agent_x, agent_y, obstacle_y]

        """
        if state is None:
            state = self.state

        obs = self.bound_in_grid(round(state[2] + np.random.normal()))

        return np.array([*state[:2], obs], dtype=int)

    def reset(self) -> np.ndarray:
        """ resets state and potentially records the episode"""

        self._state = self.sample_start_state()
        return self.generate_observation()

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """ simulates stepping from state using action. Returns interaction

        Args:
             state: (`np.ndarray`): [x, y, obs_y]
             action: (`int`): 0 is go down, 1 is stay or 2 is go up

        RETURNS (`po_nrl.environments.SimulationResult`):

        """
        assert self.action_space.contains(action), f'action {action} not in space'
        assert self.state_space.contains(state), f'state {state} not in space'
        assert state[0] > 0, f'state {state} is terminal because x = 0'

        new_state = state.copy()

        # move agent
        new_state[0] -= 1
        new_state[1] = self.bound_in_grid(
            state[1] + self.action_to_move[int(action)]
        )

        # move obstacle
        prob = np.random.uniform()
        if prob < self._block_policy[0]:
            new_state[2] -= 1
        if prob - self._block_policy[0] > self._block_policy[1]:
            new_state[2] += 1

        new_state[2] = self.bound_in_grid(new_state[2])

        # observation
        obs = self.generate_observation(new_state)

        return SimulationResult(new_state, obs)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """ 0 if staying, small penalty for going diagonal, huge cost if collision

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """

        assert self.state_space.contains(state)
        assert self.state_space.contains(new_state)
        assert self.action_space.contains(action)

        reward = 0 if action == 1 else -1

        if new_state[0] == 0 and new_state[1] == new_state[2]:
            reward += self.COLLISION_REWARD

        return reward

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """ True if reached end in `new_state`

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`): whether the transition is terminal

        """

        assert self.state_space.contains(state)
        assert self.state_space.contains(new_state)
        assert self.action_space.contains(action)

        return bool(new_state[0] == 0)

    def step(self, action: int) -> EnvironmentInteraction:
        """ updates the state and return observed transitions

        Will move the agent 1 cell to the left, and (depending on the action)
        up to 1 cell vertically. Stochastically move the obstacle,
        generating an observation in the process

        Is terminal when the agent reached the last column

        Args:
             action: (`int`): 0 is go down, 1 is stay or 2 is go up

        RETURNS (`po_nrl.environments.EnvironmentInteraction`): the transition

        """
        assert 0 <= action < 3

        sim_step = self.simulation_step(self.state, action)
        reward = self.reward(self.state, action, sim_step.state)
        terminal = self.terminal(self.state, action, sim_step.state)

        if self.log_is_on(POBNRLogger.LogLevel.V2):
            self.log(
                POBNRLogger.LogLevel.V2,
                f"Step: (x: {self.state[0]}, y: {self.state[1]}) and a="
                f"{self.action_to_string[action]} --> "
                f"(x:{sim_step.state[0]}, y: {sim_step.state[1]}),"
                " with obstacle "
                f"{sim_step.state[2]} (obs: {sim_step.observation[-1]})"
            )

        self.state = sim_step.state

        return EnvironmentInteraction(
            sim_step.observation, reward, terminal
        )

    def obs2index(self, observation: np.array) -> int:
        """ projects the observation as an int

        Args:
             observation: (`np.array`): observation to project

        RETURNS (`int`): int representation of observation

        """
        assert self.observation_space.contains(observation), \
            f"{observation} not in space {self.observation_space}"
        assert np.all(self.size > observation) and np.all(observation >= 0),\
            f"expecting all observation to be more than 0, {observation}"

        return observation[0]\
            + observation[1] * self.size\
            + observation[2] * self.size * self.size

    def __repr__(self):
        return (f' Collision avoidance of size {self.size} with '
                f' obstacle probabilities {self._block_policy}')
