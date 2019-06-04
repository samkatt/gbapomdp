""" collision avoidance environment """

import numpy as np

from environments import Environment, EnvironmentInteraction, ActionSpace
from environments import POUCTSimulator, POUCTInteraction
from misc import DiscreteSpace, POBNRLogger


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

    def __init__(self, domain_size: int):
        """ constructs a Collision Avoidance domain of specified size

        Args:
             domain_size: (`int`): the size of the grid

        """

        assert domain_size > 0, "Domain size must be > 0"
        assert domain_size % 2 == 1, "Domain size must be odd"

        POBNRLogger.__init__(self)

        self._size = domain_size
        self._mid = int(self._size / 2)

        self._action_space = ActionSpace(3)
        self._obs_space = DiscreteSpace([self._size, self._size, self._size])

        self._state = self.sample_start_state()

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

        assert self._size > state['agent_x'] >= 0
        assert self._size > state['agent_y'] >= 0
        assert self._size > state['obstacle'] >= 0

        self._state = state

    def sample_start_state(self) -> dict:
        """ returns the (deterministic) start state

        Args:

        RETURNS (`dict`): {'agent_x': `int`, 'agent_y': `int`, 'obstacle': `int` }

        """
        return {
            'agent_x': self._size - 1,
            'agent_y': self._mid,
            'obstacle': self._mid
        }

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

        self._state = self.sample_start_state()
        return self.generate_observation()

    def simulation_step(self, state: dict, action: int) -> POUCTInteraction:
        """ simulates stepping from state using action. Returns interaction

        Args:
             state: (`dict`): {'agent_x': int, 'agent_y': int, 'obstacle': int}
             action: (`int`): 0 is go down, 1 is stay or 2 is go up

        RETURNS (`pobnrl.environments.POUCTInteraction`):

        """
        assert 0 <= action < 3

        new_state = state.copy()

        # move agent
        new_state['agent_x'] -= 1
        new_state['agent_y'] = self.bound_in_grid(
            state['agent_y'] + self.action_to_move[int(action)]
        )

        # move obstacle
        if np.random.random() < self.BLOCK_MOVE_PROB:
            if np.random.random() < .5:
                new_state['obstacle'] += 1
            else:
                new_state['obstacle'] -= 1

            new_state['obstacle'] = self.bound_in_grid(new_state['obstacle'])

        # observation
        obs = self.generate_observation(new_state)

        # reward and terminal
        reward = 0 if action == 1 else -1
        terminal = False

        if new_state['agent_x'] == 0:
            terminal = True
            if new_state['agent_y'] == new_state['obstacle']:
                reward = self.COLLISION_REWARD

        return POUCTInteraction(new_state, obs, reward, terminal)

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

        if self.log_is_on(POBNRLogger.LogLevel.V3):
            self.log(
                POBNRLogger.LogLevel.V3,
                f"Step: (x: {self.state['agent_x']}, y: {self.state['agent_y']}) and a="
                f"{self.action_to_string[action]} --> "
                f"(x:{transition.state['agent_x']}, y: {transition.state['agent_y']}),"
                " with obstacle "
                f"{transition.state['obstacle']} (obs:  {transition.observation[-1]})"
            )

        self.state = transition.state

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
        assert np.all(self.size > observation) and np.all(observation >= 0),\
            f"expecting all observation to be more than 0, {observation}"

        return observation[0]\
            + observation[1] * self.size\
            + observation[2] * self.size * self.size

    @property
    def action_space(self) -> ActionSpace:
        """ a `pobnrl.environments.ActionSpace`([3]) space """
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `pobnrl.misc.DiscreteSpace`([grid_height]) space """
        return self._obs_space
