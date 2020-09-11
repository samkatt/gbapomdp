"""A wrapper for domains found in gym-gridverse package"""
import numpy as np

from gym_gridverse.envs.gridworld import GridWorld as GverseEnv
from gym_gridverse.grid_object import Goal, MovingObstacle, Wall
from po_nrl.environments import (
    ActionSpace,
    Environment,
    EnvironmentInteraction,
    SimulationResult,
    Simulator,
)
from po_nrl.misc import Space, DiscreteSpace


def object_type_index_on_agents_location(state: np.ndarray) -> int:
    """gets the object type index of the object on the location of the agent's position

    Depends on the state representation, assumes first channel in `state`
    contains type indices

    TODO: test

    Args:
        state (`np.ndarray`):

    Returns:
        `int`: the type index of the object on the agent's position
    """
    x, y = 0, 0  # TODO: nyi
    return state[y, x, 0]


def is_agent_on_goal(state: np.ndarray) -> bool:
    """returns true if agent in `state` is located on a goal

    Depends on the state representation, but in practice find the agent in the
    state and check whether its position in the grid contains a goal

    Calls `object_type_index_on_agents_location`

    Args:
        state (`np.ndarray`):

    Returns:
        `bool`: true if the agent is on the goal
    """
    return (
        object_type_index_on_agents_location(state) == Goal.type_index
    )  # pylint: disable=no-member


def has_agent_collided_into_obstacle(state: np.ndarray) -> bool:
    """returns true if agent in `state` is located on an obstacle

    Depends on the state representation, but in practice find the agent in the
    state and check whether its position in the grid contains a obstacle

    Calls `object_type_index_on_agents_location`

    Args:
        state (`np.ndarray`):

    Returns:
        `bool`: true if the agent is on an obstacle
    """
    return (
        object_type_index_on_agents_location(state) == MovingObstacle.type_index
    )  # pylint: disable=no-member


def will_agent_collide_into_wall(state: np.ndarray, action: int) -> bool:
    """returns whether `action` in `state` will lead to a collision with a wall

    Assumes walls do not move and actions are deterministic

    TODO: test

    Args:
        state (`np.ndarray`):
        action (`int`):

    Returns:
        `bool`: true if agent will run into a wall when doing `action` in `state`
    """
    intended_x, intended_y = 0, 0  # TODO: nyi
    return (
        state[intended_y, intended_x, 0] == Wall.type_index
    )  # pylint: disable=no-member


class GridverseDomain(Environment, Simulator):
    """ Wrapper around gridverse domains

    The gridverse repository can be found at
    `https://github.com/abaisero/gym-gridverse`.
    """

    def __init__(
        self, gridverse_domain: GverseEnv,
    ):
        self._gverse_env = gridverse_domain

        self._action_space = ActionSpace(
            self._gverse_env.action_space.num_actions
        )
        # TODO: nyi
        self._obs_space = DiscreteSpace([0])
        self._state_space = DiscreteSpace([0])

    def reset(self) -> np.ndarray:
        """interface"""
        self._gverse_env.reset()
        obs = np.array()  # TODO: implement some merging of the observation

        return obs

    def step(self, action: int) -> EnvironmentInteraction:
        """interface"""
        reward, terminal = self._gverse_env.step(action)
        obs = np.array()  # TODO: implement some merging of the observation

        return EnvironmentInteraction(obs, reward, terminal)

    @property
    def action_space(self) -> ActionSpace:
        """interface"""
        return self._action_space

    @property
    def observation_space(self) -> Space:
        """interface"""
        return self._obs_space

    @property
    def state_space(self) -> Space:
        """interface"""
        return self._state_space

    def simulation_step(
        self, state: np.ndarray, action: int
    ) -> SimulationResult:
        """interface"""
        raise NotImplementedError("Grid-verse will never work as a simulator")

    def sample_start_state(self) -> np.ndarray:
        """interface"""
        # TODO: implement some merging of the observation
        return self._gverse_env.functional_reset()

    def obs2index(self, observation: np.ndarray) -> int:
        """interface"""
        # TODO: nyi

    def reward(
        self, state: np.ndarray, action: int, new_state: np.ndarray
    ) -> float:
        """interface"""
        if is_agent_on_goal(new_state):
            return 1.0

        if has_agent_collided_into_obstacle(
            new_state
        ) or will_agent_collide_into_wall(state, action):
            return -1

        return 0

    def terminal(
        self, state: np.ndarray, action: int, new_state: np.ndarray
    ) -> bool:
        """interface"""
        return (
            has_agent_collided_into_obstacle(new_state)
            or is_agent_on_goal(new_state)
            or will_agent_collide_into_wall(state, action)
        )
