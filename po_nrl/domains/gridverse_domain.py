"""A wrapper for domains found in gym-gridverse package"""
import random
from typing import Dict, Tuple

import numpy as np
from gym_gridverse.actions import TRANSLATION_ACTIONS
from gym_gridverse.actions import Actions as GverseAction
from gym_gridverse.data_gen import \
    sample_transitions as sample_gverse_transitions
from gym_gridverse.envs.env import Environment as GverseEnv
from gym_gridverse.envs.factory import gym_minigrid_from_descr
from gym_gridverse.geometry import Orientation
from gym_gridverse.geometry import Position as GversePosition
from gym_gridverse.grid_object import Goal, MovingObstacle, Wall
from gym_gridverse.observation import Observation as GverseObs
from gym_gridverse.representations.observation_representations import \
    DefaultObservationRepresentation
from gym_gridverse.representations.state_representations import \
    DefaultStateRepresentation
from gym_gridverse.simulator import Simulator as GverseSimulator
from gym_gridverse.state import State as GverseState
from po_nrl.environments import (ActionSpace, Environment,
                                 EnvironmentInteraction, SimulationResult,
                                 Simulator)
from po_nrl.misc import DiscreteSpace, POBNRLogger, Space


def flatten_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    """Represents a state by concatenating its values

    The return value for state representations in grid verse is a dictionary of
    string -> numpy arrays. This is the easiest way of converting such
    representations into a single array

    Note that this implementation keeps only the 'item index' and ignores the
    other layers. This because it is assumed that all the domains currently
    used hold no information in those layers

    Args:
        state_or_obs (`Dict[str, np.ndarray]`): output of 'representation' in gridverse

    Returns:
        `np.ndarray`: representation of input into single array
    """
    one_hot_orientation = np.zeros(4, dtype=int)
    one_hot_orientation[state['agent'][2]] = 1

    position = state['agent'][:2]

    return np.concatenate(
        [state['grid'][:, :, 3].flatten(), position, one_hot_orientation]
    )


def flatten_observation(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Represents an observation by concatenating its values

    The return value for observation representations in grid verse is a
    dictionary of string -> numpy arrays. This is the easiest way of converting
    such representations into a single array

    This function returns the grid part (obs['grid']) and flattens it.

    Note that this implementation keeps only the 'item index' and ignores the
    other layers. This because it is assumed that all the domains currently
    used hold no information in those layers

    Args:
        obs (`Dict[str, np.ndarray]`): output of 'representation' in gridverse

    Returns:
        `np.ndarray`: representation of input into single array
    """
    return obs['grid'][:, :, 3].flatten()


def reshape_state(
    flat_state: np.ndarray, h: int, w: int
) -> Dict[str, np.ndarray]:
    """Reshapes a flattened state back

    Basically reverse of `flatten_state`: reverts a flat state
    representation into semantically useful things

    TODO: could probably return something more... Comprehensible

    Args:
        flat_state (`np.ndarray`): flat (?,) vector array representing state
        h (`int`): height of grid
        w (`int`): width of grid

    Returns:
        `Dict[str, np.ndarray]`: mapping from string to semantically useful numpy arrays
                                 dictionary contains {'grid', 'agent'}

    """

    # hard coded knowledge: the last six elements describe the agent position
    # and orientation
    grid = flat_state[:-6].reshape((h, w))
    agent_pos = flat_state[-6:-4]
    agent_orientation = np.argmax(flat_state[-4:])

    return {
        'grid': grid,
        'agent': np.concatenate([agent_pos, [agent_orientation]]),
    }


class GridverseDomain(Environment, Simulator, POBNRLogger):
    """ Wrapper around gridverse domains

    The gridverse repository can be found at
    `https://github.com/abaisero/gym-gridverse`.

    Limited to plain goal-seeking and moving obstacle domains.

    - limited actions to turning and moving (excluding pick/drop and execution
    - only consider the type-index layer

    """

    def __init__(
        self,
        gverse_env: GverseEnv = gym_minigrid_from_descr(
            "MiniGrid-Empty-5x5-v0"
        ),
    ):
        """The only input to our grid-verse wrapper is the actual gverse domain itself

        Our class wraps the domain basically twice: it also wraps it into the
        Gridverse simulator, which provides functionality to convert abstract
        state and observations into numpy arrays.

        This is basically just an interface between the grid verse simulator
        and our code. Its biggest responsibility is translating the dictionary
        of numpy arrays generated by gridverse into a single array (see
        `flatten_state_or_observation` and `reshape_state`).

        Args:
            gverse_env (`GverseEnv, optional`):
        """
        POBNRLogger.__init__(self)

        state_rep = DefaultStateRepresentation(gverse_env.state_space)
        obs_rep = DefaultObservationRepresentation(gverse_env.observation_space)

        self._gverse_sim = GverseSimulator(gverse_env, state_rep, obs_rep)

        self.h = gverse_env.state_space.grid_shape.height
        self.w = gverse_env.state_space.grid_shape.width
        self.obs_h = gverse_env.observation_space.grid_shape.height
        self.obs_w = gverse_env.observation_space.grid_shape.width

        self._action_space = ActionSpace(6)

        self._obs_space = DiscreteSpace(
            # pylint: disable=no-member
            [self._gverse_sim.env.state_space.max_grid_object_type + 1]
            * self.obs_h
            * self.obs_w
        )
        self._state_space = DiscreteSpace(
            # pylint: disable=no-member
            [self._gverse_sim.env.state_space.max_grid_object_type + 1]
            * self.h
            * self.w
            + [self.h, self.w]
            + [2] * len(Orientation)
        )

    def _convert_gverse_state(self, s: GverseState) -> np.ndarray:
        return flatten_state(self._gverse_sim.state_rep.convert(s))

    def _convert_gverse_obs(self, o: GverseObs) -> np.ndarray:
        return flatten_observation(self._gverse_sim.obs_rep.convert(o))

    def reset(self) -> np.ndarray:
        """interface"""
        self._gverse_sim.env.reset()
        return self._convert_gverse_obs(self._gverse_sim.env.observation)

    def step(self, action: int) -> EnvironmentInteraction:
        """interface"""
        a = GverseAction(action)
        reward, terminal = self._gverse_sim.env.step(a)
        obs = self._convert_gverse_obs(self._gverse_sim.env.observation)

        if self.log_is_on(POBNRLogger.LogLevel.V2):
            self.log(
                POBNRLogger.LogLevel.V2,
                f"Env: after a={a} agent on {self._gverse_sim.env.state.agent.position}",
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
        return self._convert_gverse_state(
            self._gverse_sim.env.functional_reset()
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

        unpacked_state = reshape_state(new_state, self.h, self.w)
        y, x = unpacked_state['agent'][0], unpacked_state['agent'][1]
        item_under_agent = unpacked_state['grid'][y, x]

        if item_under_agent == Goal.type_index:
            return 1.0

        if item_under_agent in [MovingObstacle.type_index, Wall.type_index]:
            return -1

        unpacked_state = reshape_state(state, self.h, self.w)
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

        unpacked_state = reshape_state(new_state, self.h, self.w)
        y, x = unpacked_state['agent'][0], unpacked_state['agent'][1]
        item_under_agent = unpacked_state['grid'][y, x]

        unpacked_state = reshape_state(state, self.h, self.w)
        prev_y, prev_x = unpacked_state['agent'][0], unpacked_state['agent'][1]

        same_position = y == prev_y and x == prev_x
        translating = action in TRANSLATION_ACTIONS

        # pylint: disable=no-member
        return item_under_agent in [
            Goal.type_index,
            Wall.type_index,
            MovingObstacle.type_index,
        ] or (translating and same_position)

    def sample_transition(
        self,
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """Samples a random transition in the domain

        First samples a state and action:

        - the state from the 'reset' function (which we assume generates all
          reachable states!)
            - explicitly set the position of the agent randomly
        - the action is sampled randomly from _our_ action space

        Then the next state and observation is sampled from the actual
        dynamics.

        Returns:
            `Tuple[np.ndarray, int, np.ndarray, np.ndarray]`: state-action-state-observation
        """

        def action_sampler(_):
            return GverseAction(self.action_space.sample())

        def state_sampler():
            s = self._gverse_sim.env.functional_reset()

            # set random agent position, it is assumed here that (apart from
            # the agent position) all other reachable states are sampled
            # through the functional reset
            s.agent.position = GversePosition(
                random.randint(1, self.h - 1), random.randint(1, self.w - 1)
            )

            return s

        gverse_transition = sample_gverse_transitions(
            1, state_sampler, action_sampler, self._gverse_sim
        )[0]

        # TODO: this does not generalize to different representations
        s = flatten_state(gverse_transition.state)
        a = gverse_transition.action.value
        ss = flatten_state(gverse_transition.next_state)
        o = flatten_observation(gverse_transition.obs)

        return (s, a, ss, o)
