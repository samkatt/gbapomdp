"""A wrapper for domains found in gym-gridverse package"""
from typing import Dict

import numpy as np
from gym_gridverse.actions import TRANSLATION_ACTIONS
from gym_gridverse.actions import Actions as GverseAction
from gym_gridverse.envs.gridworld import GridWorld as GverseEnv
from gym_gridverse.grid_object import Goal, MovingObstacle, Wall
from gym_gridverse.representations.observation_representations import \
    DefaultObservationRepresentation
from gym_gridverse.representations.state_representations import \
    DefaultStateRepresentation
from po_nrl.environments import (ActionSpace, Environment,
                                 EnvironmentInteraction, SimulationResult,
                                 Simulator)
from po_nrl.misc import DiscreteSpace, Space


def flatten_state_or_observation(
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
    return np.concatenate(
        [state_or_obs['grid'][:, :, 3].flatten(), state_or_obs['agent'][:-3]]
    )


def reshape_state_or_observation(
    flat_state_or_obs: np.ndarray, h: int, w: int
) -> Dict[str, np.ndarray]:
    """Reshapes a flattened state or observation back

    Basically reverse of `flatten_state_or_observation`: reverts a flat
    state/observation representation into semantically useful things

    Args:
        flat_state_or_obs (`np.ndarray`): flat (?,) vector array representing state or observation
        h (`int`): height of grid
        w (`int`): width of grid

    Returns:
        `Dict[str, np.ndarray]`: mapping from string to semantically useful numpy arrays
                                 dictionary contains {'grid', 'agent'}

    """

    # hard coded knowledge: the last three elements describe the held item
    num_features = h * w

    grid = flat_state_or_obs[:num_features].reshape((h, w))
    agent = flat_state_or_obs[num_features:]

    return {
        'grid': grid,
        'agent': agent,
    }


class GridverseDomain(Environment, Simulator):
    """ Wrapper around gridverse domains

    The gridverse repository can be found at
    `https://github.com/abaisero/gym-gridverse`.

    Limited to plain goal-seeking and moving obstacle domains.

    - limited actions to turning and moving (excluding pick/drop and execution
    - only consider the type-index layer

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

        self._action_space = ActionSpace(6)

        # TODO: nyi
        self._obs_space = DiscreteSpace([])
        self._state_space = DiscreteSpace([])
        self.h, self.w = 7, 7

    def reset(self) -> np.ndarray:
        """interface"""
        self._gverse_env.reset()
        return flatten_state_or_observation(
            self._obs_rep.convert(self._gverse_env.observation)
        )

    def step(self, action: int) -> EnvironmentInteraction:
        """interface"""
        reward, terminal = self._gverse_env.step(GverseAction(action))
        obs = flatten_state_or_observation(
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
        return flatten_state_or_observation(
            self._state_rep.convert(self._gverse_env.functional_reset())
        )

    def reward(
        self, state: np.ndarray, action: int, new_state: np.ndarray
    ) -> float:
        """interface

        -1 if:
            - agent is on a wall or moving obstacle in new state
            - agent 'moved' but stayed on the same location
        1 if:
            - agent is on a goal in new state
        else 0
        """

        # pylint: disable=no-member

        unpacked_state = reshape_state_or_observation(new_state, self.h, self.w)
        y, x = unpacked_state['agent'][0], unpacked_state['agent'][1]
        item_under_agent = unpacked_state['grid'][y, x]

        if item_under_agent == Goal.type_index:
            return 1.0

        if item_under_agent in [MovingObstacle.type_index, Wall.type_index]:
            return -1

        unpacked_state = reshape_state_or_observation(new_state, self.h, self.w)
        prev_y, prev_x = unpacked_state['agent'][0], unpacked_state['agent'][1]

        same_position = y == prev_y and x == prev_x
        translating = action in TRANSLATION_ACTIONS

        if translating and same_position:
            return -1

        return 0

    def terminal(
        self, state: np.ndarray, action: int, new_state: np.ndarray
    ) -> bool:
        """interface

        True if:
            - agent is on a wall, goal or moving obstacle in new state
            - agent 'moved' but stayed on the same location
        """

        unpacked_state = reshape_state_or_observation(new_state, self.h, self.w)
        y, x = unpacked_state['agent'][0], unpacked_state['agent'][1]
        item_under_agent = unpacked_state['grid'][y, x]

        unpacked_state = reshape_state_or_observation(new_state, self.h, self.w)
        prev_y, prev_x = unpacked_state['agent'][0], unpacked_state['agent'][1]

        same_position = y == prev_y and x == prev_x
        translating = action in TRANSLATION_ACTIONS

        # pylint: disable=no-member
        return item_under_agent in [
            Goal.type_index,
            Wall.type_index,
            MovingObstacle.type_index,
        ] or (translating and same_position)
