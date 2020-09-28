"""A wrapper for domains found in gym-gridverse package"""
import abc
import random
from typing import Tuple

import numpy as np
from gym_gridverse.actions import TRANSLATION_ACTIONS
from gym_gridverse.actions import Actions as GverseAction
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
from gym_gridverse.state import State as GverseState
from po_nrl.environments import (ActionSpace, Environment,
                                 EnvironmentInteraction, SimulationResult,
                                 Simulator)
from po_nrl.misc import DiscreteSpace, POBNRLogger, Space


class StateEncoding(abc.ABC):
    """contains 'encoding' and 'decoding' paired functionality"""

    @abc.abstractmethod
    def encode(self, s: GverseState) -> np.ndarray:
        """encodes a state in Gridverse into a single numpy array

        Args:
            s (`GverseState`): the state to be encoded

        Returns:
            `np.ndarray:` representation of the state as single array
        """

    @abc.abstractmethod
    def decode(
        self, s: np.ndarray
    ) -> Tuple[np.ndarray, GversePosition, Orientation]:
        """decodes a state-array into semantic meaning

        Specifically extracts the grid as a (h x w) grid, the position of the
        agent and its orientation

        Args:
            s (`np.ndarray`): state as encoded by `encode`

        Returns:
            `Tuple[np.ndarray, GversePosition, Orientation]`: semantic meaning of state: grid (h x w), y, x
        """

    @property
    @abc.abstractmethod
    def state_space(self) -> DiscreteSpace:
        """since encoding determines the state space, this functionality belongs here"""


class CompactStateEncoding(StateEncoding):
    """default encoding, no one-hot encoding applied"""

    def __init__(self, rep: DefaultStateRepresentation):
        self._rep = rep

        h, w = self._rep.state_space.grid_shape
        self._state_space = DiscreteSpace(
            # pylint: disable=no-member
            [self._rep.state_space.max_grid_object_type + 1] * h * w
            + [h, w, len(Orientation)]
        )

    def encode(self, s: GverseState) -> np.ndarray:
        """encodes `s` into a compact numpy array

        From `s` the grid type indices are taken and flattened, ontop of which
        the position and orientation is concatenated (as is)

        Args:
            s (`GverseState`):

        Returns:
            `np.ndarray`:
        """
        state = self._rep.convert(s)

        return np.concatenate(
            [state['grid'][:, :, 3].flatten(), state['agent'][:3]]
        )

    def decode(
        self, s: np.ndarray
    ) -> Tuple[np.ndarray, GversePosition, Orientation]:
        """decodes a numpy array (generated from `encode`) and extracts info

        Returns (grid (h x w), position (y, x), orientation)

        Args:
            s (`np.ndarray`):

        Returns:
            `Tuple[np.ndarray, GversePosition, Orientation]`:
        """
        h, w = self._rep.state_space.grid_shape
        num_grid_dim = h * w

        return (
            s[:num_grid_dim].reshape(h, w),
            GversePosition(*s[-3:-1]),
            Orientation(s[-1]),
        )

    @property
    def state_space(self) -> DiscreteSpace:
        return self._state_space


class OneHotOrientationEncoding(StateEncoding):
    """one-hot encoding of only orientation, not of position"""

    def __init__(self, rep: DefaultStateRepresentation):
        self._rep = rep

        h, w = self._rep.state_space.grid_shape

        self._state_space = DiscreteSpace(
            # pylint: disable=no-member
            [self._rep.state_space.max_grid_object_type + 1] * h * w
            + [h, w, 2, 2, 2, 2]
        )

    def encode(self, s: GverseState) -> np.ndarray:
        """encodes the orientation as one-hot

        Otherwise just concatenates flattened grid and positions

        Args:
            s (`GverseState`):

        Returns:
            `np.ndarray`:
        """
        state = self._rep.convert(s)

        one_hot_orientation = np.zeros(len(Orientation))
        one_hot_orientation[s.agent.orientation.value] = 1

        return np.concatenate(
            [
                state['grid'][:, :, 3].flatten(),
                list(s.agent.position),
                one_hot_orientation,
            ]
        )

    def decode(
        self, s: np.ndarray
    ) -> Tuple[np.ndarray, GversePosition, Orientation]:
        """interprets encoded `s` (from `encode`) and return info

        Basically interprets hot-encoding of orientation

        Args:
            s (`np.ndarray`):

        Returns:
            `Tuple[np.ndarray, GversePosition, Orientation]`:
        """
        h, w = self._rep.state_space.grid_shape
        grid_ndim = h * w

        orientation = np.argmax(s[-len(Orientation):])

        return (
            s[:grid_ndim].reshape(h, w),
            # 2 positions
            GversePosition(*s[grid_ndim: grid_ndim + 2]),
            Orientation(orientation),
        )

    @property
    def state_space(self) -> DiscreteSpace:
        return self._state_space


class OneHotStateEncoding(StateEncoding):
    """one-hot encoding for both position and orientation of the agent"""

    def __init__(self, rep: DefaultStateRepresentation):
        self._rep = rep

        h, w = self._rep.state_space.grid_shape

        self._state_space = DiscreteSpace(
            # pylint: disable=no-member
            [self._rep.state_space.max_grid_object_type + 1] * h * w
            # one-hot encoding of position and orientation
            + [2] * (h + w + len(Orientation))
        )

    def encode(self, s: GverseState) -> np.ndarray:
        """Represents a state as a numpy array, hot encoding position and orientation

        Flattens the 'grid' and concatenates a one-hot encoding of the agent's
        position and orientation

        Note that this implementation keeps only the 'item index' and ignores the
        other layers. This because it is assumed that all the domains currently
        used hold no information in those layers

        Args:
            s (`GverseState`): state in Gridverse

        Returns:
            `np.ndarray`: representation of input into single array
        """
        state = self._rep.convert(s)

        h, w = state['grid'].shape[:2]
        y, x = state['agent'][:2]
        orientation = state['agent'][2]

        one_hot_position = np.zeros(h + w, dtype=int)
        one_hot_position[y] = 1
        one_hot_position[h + x] = 1

        one_hot_orientation = np.zeros(4, dtype=int)
        one_hot_orientation[orientation] = 1

        return np.concatenate(
            [
                state['grid'][:, :, 3].flatten(),
                one_hot_position,
                one_hot_orientation,
            ]
        )

    def decode(
        self, s: np.ndarray
    ) -> Tuple[np.ndarray, GversePosition, Orientation]:
        """reshapes a flattened state back into semantic info

        Assumes one-hot encoded state as implemented in `encode` and extracts the grid and position & orientation of the agent

        Args:
            s (`np.ndarray`): flat (?,) vector array representing state

        Returns:
            `Tuple[np.ndarray, GversePosition, Orientation]`:

        """
        h, w = self._rep.state_space.grid_shape

        num_grid_dim = h * w
        num_orientation_dim = 4

        grid = s[:num_grid_dim].reshape((h, w))

        # decode one-hot encoding of the position
        agent_y = np.argmax(s[num_grid_dim: num_grid_dim + h])
        agent_x = np.argmax(s[num_grid_dim + h: num_grid_dim + h + w])

        # decode one-hot encoding of the orientation
        agent_orientation = np.argmax(s[-num_orientation_dim:])

        return (
            grid,
            GversePosition(agent_y, agent_x),
            Orientation(agent_orientation),
        )

    @property
    def state_space(self) -> DiscreteSpace:
        return self._state_space


def flatten_observation(
    obs: GverseObs, rep: DefaultObservationRepresentation
) -> np.ndarray:
    """Represents an `obs` by calling `rep` and concatenating grid

    The return value of `rep` on an `obs` in grid verse is a dictionary of
    string -> numpy arrays. This is the easiest way of converting such
    representations into a single array

    This function returns the grid part (obs['grid']) and flattens it.

    Note that this implementation keeps only the 'item index' and ignores the
    other layers. This because it is assumed that all the domains currently
    used hold no information in those layers

    Args:
        obs (`GverseObs`): observation in Gridverse
        rep (`DefaultObservationRepresentation`): representation of `obs` into multiply arrays

    Returns:
        `np.ndarray`: representation of input into single array
    """
    return rep.convert(obs)['grid'][:, :, 3].flatten()


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

        self._gverse_obs_rep = DefaultObservationRepresentation(
            gverse_env.observation_space
        )

        self._gverse_env = gverse_env

        self._state_encoding = OneHotStateEncoding(
            DefaultStateRepresentation(gverse_env.state_space)
        )

        self.h = gverse_env.state_space.grid_shape.height
        self.w = gverse_env.state_space.grid_shape.width
        self.obs_h = gverse_env.observation_space.grid_shape.height
        self.obs_w = gverse_env.observation_space.grid_shape.width

        self._action_space = ActionSpace(6)

        self._obs_space = DiscreteSpace(
            # pylint: disable=no-member
            [self._gverse_env.state_space.max_grid_object_type + 1]
            * self.obs_h
            * self.obs_w
        )

    def _convert_gverse_obs(self, o: GverseObs) -> np.ndarray:
        return flatten_observation(o, self._gverse_obs_rep)

    def reset(self) -> np.ndarray:
        """interface"""
        self._gverse_env.reset()
        return self._convert_gverse_obs(self._gverse_env.observation)

    def step(self, action: int) -> EnvironmentInteraction:
        """interface"""
        a = GverseAction(action)
        reward, terminal = self._gverse_env.step(a)
        obs = self._convert_gverse_obs(self._gverse_env.observation)

        if self.log_is_on(POBNRLogger.LogLevel.V2):
            self.log(
                POBNRLogger.LogLevel.V2,
                f"Env: after a={a} agent on {self._gverse_env.state.agent.position} facing {self._gverse_env.state.agent.orientation}",
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
        return self._state_encoding.state_space

    def simulation_step(
        self, state: np.ndarray, action: int
    ) -> SimulationResult:
        """interface"""
        raise NotImplementedError("Grid-verse will never work as a simulator")

    def sample_start_state(self) -> np.ndarray:
        """interface"""
        return self._state_encoding.encode(self._gverse_env.functional_reset())

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

        grid, pos, _ = self._state_encoding.decode(new_state)

        y, x = pos
        item_under_agent = grid[y, x]

        if item_under_agent == Goal.type_index:
            return 1.0

        if item_under_agent in [MovingObstacle.type_index, Wall.type_index]:
            return -1

        grid, pos, _ = self._state_encoding.decode(state)
        prev_y, prev_x = pos

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

        grid, pos, _ = self._state_encoding.decode(new_state)
        y, x = pos
        item_under_agent = grid[y, x]

        grid, pos, _ = self._state_encoding.decode(state)
        prev_y, prev_x = pos

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

        a = GverseAction(self.action_space.sample())
        s = self._gverse_env.functional_reset()

        # set random agent position, it is assumed here that (apart from
        # the agent position) all other reachable states are sampled
        # through the functional reset
        s.agent.position = GversePosition(
            random.randint(1, self.h - 1), random.randint(1, self.w - 1)
        )

        next_s, _, _ = self._gverse_env.functional_step(s, a)
        o = self._gverse_env.functional_observation(next_s)

        return (
            self._state_encoding.encode(s),
            a.value,
            self._state_encoding.encode(next_s),
            self._convert_gverse_obs(o),
        )
