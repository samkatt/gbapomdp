"""A wrapper for domains found in gym-gridverse package"""
from typing import Dict, List

import numpy as np
from gym_gridverse.envs.gridworld import GridWorld as GverseEnv
from gym_gridverse.grid_object import Goal, MovingObstacle, Wall
from gym_gridverse.representations.observation_representations import \
    DefaultObservationRepresentation
from gym_gridverse.representations.state_representations import \
    DefaultStateRepresentation
from gym_gridverse.state import State as GverseState
from po_nrl.environments import (ActionSpace, Environment,
                                 EnvironmentInteraction, SimulationResult,
                                 Simulator)
from po_nrl.misc import DiscreteSpace, Space


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
        # pylint: disable=no-member
        object_type_index_on_agents_location(state)
        == Goal.type_index
    )


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
        # pylint: disable=no-member
        object_type_index_on_agents_location(state)
        == MovingObstacle.type_index
    )


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
        # pylint: disable=no-member
        state[intended_y, intended_x, 0]
        == Wall.type_index
    )


def unpack_gridverse_state(s: GverseState) -> Dict[str, np.ndarray]:
    """rigorous way of extracting information from the gridverse state

    Will return basically whatever you will need to know in convenient shape as
    different entries in the returned dictionary

    Return:
        'agent_pos' -> [x,y] `np.ndarray`
        'agent_dir' -> [direction] `np.ndarray`
        'obstacles' -> [[x_1,y_1] ... [x_n, y_n]] `np.ndarray`
        'agent_pos' -> [x,y] `np.ndarray`
        'array' -> concatenation of above as vector

    Assumes grid only contains moving obstacles

    TODO:  use and test or remove

    Args:
        s (`GverseState`):

    Returns:
        `Dict[str, np.ndarray]`:
    """

    obstacles: List[List[int]] = []
    for pos in s.grid.positions():
        # pylint: disable=no-member
        if s.grid[pos].type_index == MovingObstacle.type_index:
            obstacles.append(*pos)

    agent_pos = np.array([s.agent.position.x, s.agent.position.y])
    agent_dir = np.array([s.agent.orientation.value])

    return {
        'agent_pos': agent_pos,
        'agent_dir': agent_dir,
        'obstacles': np.array(obstacles),
        'array': np.concatenate(
            [agent_pos, agent_dir, np.array(obstacles).flatten()]
        ),
    }


class GridverseDomain(Environment, Simulator):
    """ Wrapper around gridverse domains

    The gridverse repository can be found at
    `https://github.com/abaisero/gym-gridverse`.
    """

    def __init__(
        self, gridverse_domain: GverseEnv,
    ):
        self._gverse_env = gridverse_domain
        self._state_rep = DefaultStateRepresentation(
            self._gverse_env.state_space
        )
        self._obs_rep = DefaultObservationRepresentation(
            self._gverse_env.observation_space
        )

        self._action_space = ActionSpace(
            self._gverse_env.action_space.num_actions
        )

        # TODO: nyi
        self._obs_space = DiscreteSpace([])
        self._state_space = DiscreteSpace([])

    @staticmethod
    def _concatenate_state_or_observation(
        state_or_obs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Represents a state or observation by concatenating its values

        The return value for state or observation representations in grid verse
        is a dictionary of string -> numpy arrays. This is the easiest way of
        converting such representations into a single array

        Args:
            state_or_obs (`Dict[str, np.ndarray]`): output of 'representation' in gridverse

        Returns:
            `np.ndarray`: representation of input into single array
        """
        return np.concatenate([x.flatten() for x in state_or_obs.values()])

    def reset(self) -> np.ndarray:
        """interface"""
        self._gverse_env.reset()
        return self._concatenate_state_or_observation(
            self._obs_rep.convert(self._gverse_env.observation)
        )

    def step(self, action: int) -> EnvironmentInteraction:
        """interface"""
        reward, terminal = self._gverse_env.step(action)
        obs = self._concatenate_state_or_observation(
            self._obs_rep.convert(self._gverse_env.observation)
        )

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
        return self._concatenate_state_or_observation(
            self._gverse_env.functional_reset()
        )

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
