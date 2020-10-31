"""A wrapper for domains found in gym-gridverse package"""
import abc
import random
from typing import List, Tuple

import numpy as np
from gym_gridverse.actions import TRANSLATION_ACTIONS
from gym_gridverse.actions import Actions as GverseAction
from gym_gridverse.envs.factory import gym_minigrid_from_descr
from gym_gridverse.geometry import Orientation
from gym_gridverse.geometry import Position as GversePosition
from gym_gridverse.grid_object import Goal, Hidden, MovingObstacle, Wall
from gym_gridverse.observation import Observation as GverseObs
from gym_gridverse.representations.observation_representations import \
    DefaultObservationRepresentation
from gym_gridverse.representations.state_representations import \
    DefaultStateRepresentation
from gym_gridverse.state import State as GverseState
from po_nrl.agents.neural_networks.neural_pomdps import DynamicsModel
from po_nrl.environments import (ActionSpace, Environment,
                                 EnvironmentInteraction, SimulationResult,
                                 Simulator)
from po_nrl.misc import DiscreteSpace, POBNRLogger, Space


class StateEncoding(abc.ABC):
    """contains 'encoding' and 'decoding' paired functionality"""

    @property
    @abc.abstractmethod
    def grid_size(self) -> int:
        """returns the size of the (square) grid

        Returns:
            int: size
        """

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

    @staticmethod
    def construct(
        descr: str, rep: DefaultStateRepresentation
    ) -> 'StateEncoding':
        """factory construction function

        expects descr to be either: 'compact', 'one-hot-state' or 'one-hot-orientation'

        Args:
            descr (`str`): string representation of the encoding
            rep (`DefaultStateRepresentation`):

        Returns:
            'StateEncoding':
        """

        if descr == "compact":
            return CompactStateEncoding(rep)
        if descr == "one-hot-orientation":
            return OneHotOrientationEncoding(rep)
        if descr == "one-hot-state":
            return OneHotStateEncoding(rep)

        raise ValueError(f"cannot map {descr} to an encoding")


class CompactStateEncoding(StateEncoding):
    """default encoding, no one - hot encoding applied"""

    def __init__(self, rep: DefaultStateRepresentation):
        self._rep = rep

        h, w = self._rep.state_space.grid_shape
        self._state_space = DiscreteSpace(
            # pylint: disable=no-member
            [self._rep.state_space.max_grid_object_type + 1] * h * w
            + [h, w, len(Orientation)]
        )

        assert h == w, f"Expecting square grid, not {h} by {w}"
        self._grid_size = h

    @property
    def grid_size(self) -> int:
        return self._grid_size

    def encode(self, s: GverseState) -> np.ndarray:
        """encodes `s` into a compact numpy array

        From `s` the grid type indices are taken and flattened, ontop of which
        the position and orientation is concatenated (as is)

        Args:
            s(`GverseState`):

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
        """decodes a numpy array(generated from `encode`) and extracts info

        Returns(grid(h x w), position(y, x), orientation)

        Args:
            s(`np.ndarray`):

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
    """one - hot encoding of only orientation, not of position"""

    def __init__(self, rep: DefaultStateRepresentation):
        self._rep = rep

        h, w = self._rep.state_space.grid_shape

        self._state_space = DiscreteSpace(
            # pylint: disable=no-member
            [self._rep.state_space.max_grid_object_type + 1] * h * w
            + [h, w, 2, 2, 2, 2]
        )

        assert h == w, f"Expecting square grid, not {h} by {w}"
        self._grid_size = h

    @property
    def grid_size(self) -> int:
        return self._grid_size

    def encode(self, s: GverseState) -> np.ndarray:
        """encodes the orientation as one - hot

        Otherwise just concatenates flattened grid and positions

        Args:
            s(`GverseState`):

        Returns:
            `np.ndarray`:
        """
        state = self._rep.convert(s)

        one_hot_orientation = np.zeros(len(Orientation), dtype=int)
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

        Basically interprets hot - encoding of orientation

        Args:
            s(`np.ndarray`):

        Returns:
            `Tuple[np.ndarray, GversePosition, Orientation]`:
        """
        h, w = self._rep.state_space.grid_shape
        grid_ndim = h * w

        # NOTE: assumes state is _legal_ and this is a proper 1-one encoding
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
    """one - hot encoding for both position and orientation of the agent"""

    def __init__(self, rep: DefaultStateRepresentation):
        self._rep = rep

        h, w = self._rep.state_space.grid_shape

        self._state_space = DiscreteSpace(
            # pylint: disable=no-member
            [self._rep.state_space.max_grid_object_type + 1] * h * w
            # one-hot encoding of position and orientation
            + [2] * (h + w + len(Orientation))
        )

        assert h == w, f"Expecting square grid, not {h} by {w}"
        self._grid_size = h

    @property
    def grid_size(self) -> int:
        return self._grid_size

    def encode(self, s: GverseState) -> np.ndarray:
        """Represents a state as a numpy array, hot encoding position and orientation

        Flattens the 'grid' and concatenates a one - hot encoding of the agent's
        position and orientation

        Note that this implementation keeps only the 'item index' and ignores the
        other layers. This because it is assumed that all the domains currently
        used hold no information in those layers

        Args:
            s(`GverseState`): state in Gridverse

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

        Assumes one - hot encoded state as implemented in `encode` and extracts the grid and position & orientation of the agent

        Args:
            s(`np.ndarray`): flat(?,) vector array representing state

        Returns:
            `Tuple[np.ndarray, GversePosition, Orientation]`:

        """
        h, w = self._rep.state_space.grid_shape

        num_grid_dim = h * w
        num_orientation_dim = 4

        grid = s[:num_grid_dim].reshape((h, w))

        # decode one-hot encoding of the position
        # NOTE: assumes state is _legal_ and this is a proper 1-one encoding
        agent_y = np.argmax(s[num_grid_dim: num_grid_dim + h])
        agent_x = np.argmax(s[num_grid_dim + h: num_grid_dim + h + w])

        # decode one-hot encoding of the orientation
        # NOTE: assumes state is _legal_ and this is a proper 1-one encoding
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

    This function returns the grid part(obs['grid']) and flattens it.

    Note that this implementation keeps only the 'item index' and ignores the
    other layers. This because it is assumed that all the domains currently
    used hold no information in those layers

    Args:
        obs(`GverseObs`): observation in Gridverse
        rep(`DefaultObservationRepresentation`): representation of `obs` into multiply arrays

    Returns:
        `np.ndarray`: representation of input into single array
    """
    return rep.convert(obs)['grid'][:, :, 3].flatten()


class ObservationModel(DynamicsModel.O):
    """known observation model for Gridverse to use by `DynamicsModel`"""

    def __init__(
        self, obs_size: int, encoding: StateEncoding, max_item_index: int
    ):
        assert obs_size % 2 == 1, f"Assume odd size, not {obs_size}"
        self._size = obs_size
        self._enc = encoding
        self._max_item_index = max_item_index

    @property
    def _half_size(self) -> int:
        return self._size // 2

    def model(
        self, state: np.ndarray, action: int, next_state: np.ndarray
    ) -> List[np.ndarray]:
        """`DynamicsModel.O` interface

        The observation model is deterministic, and here we simply call
        `self.sample(...)` to determine the true observation, and make sure the
        model predicts this observation with 100%
        """

        deterministic_observation = self.sample(state, action, next_state, 1)
        ndim = len(deterministic_observation)

        observation_model = np.zeros((ndim, self._max_item_index))

        observation_model[np.arange(ndim), deterministic_observation] = 1

        return observation_model

    def sample(
        self, state: np.ndarray, action: int, next_state: np.ndarray, num: int,
    ) -> np.ndarray:
        """`DynamicsModel.O` interface

        Gridverse has deterministic observations. They basically are a cropped
        and rotated version of the POV of the agent. This function does exactly
        that: returns the visible area from the agent's POV
        """

        # the observation is a fairly straightforward: take the grid of the
        # next state and crop + orient it from the agent's perspective
        grid, (y, x), orient = self._enc.decode(next_state)

        _, w = grid.shape

        ret = (
            np.ones((self._size, self._size), dtype=int)
            * Hidden.type_index  # pylint: disable=no-member
        )

        # rotate grid and position to pretend we are facing north to simplify
        # the rest of the computation
        grid, y, x = self.rotate(grid, y, x, orient)

        # get the respective indices of the grids, assuming orientation is north
        ymin_in_grid = max(0, y - self._size + 1)
        ymin_in_obs = max(0, self._size - y - 1)

        xmin_in_grid = max(0, x - self._half_size)
        xmin_in_obs = max(0, self._half_size - x)

        xmax_in_grid = min(w - 1, x + self._half_size)
        xmax_in_obs = min(self._size, self._half_size + (w - x)) - 1

        # complete the mapping by filling in the observation
        ret[ymin_in_obs:, xmin_in_obs: xmax_in_obs + 1] = grid[
            ymin_in_grid: y + 1, xmin_in_grid: xmax_in_grid + 1
        ]

        return ret.flatten()

    @staticmethod
    def rotate(
        grid: np.ndarray, y: int, x: int, orient: Orientation
    ) -> Tuple[np.ndarray, int, int]:
        """rotates a `grid` and (`y`, `x`) position according to `orient`

        Args:
            grid (`np.ndarray`):
            y (`int`):
            x (`int`):
            orient (`Orientation`):

        Returns:
            `Tuple[np.ndarray, int, int]`:
        """
        h, w = grid.shape
        rotations = {
            Orientation.N: 0,
            Orientation.E: 1,
            Orientation.S: 2,
            Orientation.W: 3,
        }

        grid = np.rot90(grid, rotations[orient])

        if orient == Orientation.N:
            new_y, new_x = y, x
        if orient == Orientation.S:
            new_y, new_x = h - y - 1, w - x - 1
        if orient == Orientation.E:
            new_y, new_x = w - x - 1, y
        if orient == Orientation.W:
            new_y, new_x = x, h - y - 1

        return grid, new_y, new_x


class GridverseDomain(Environment, Simulator, POBNRLogger):
    """ Wrapper around gridverse domains

    The gridverse repository can be found at
    `https: // github.com / abaisero / gym - gridverse`.

    Limited to plain goal - seeking and moving obstacle domains.

    - limited actions to turning and moving(excluding pick / drop and execution
    - only consider the type - index layer

    """

    def __init__(
        self,
        encoding_descr: str = "compact",
        env_descr: str = "MiniGrid-Empty-5x5-v0",
    ):
        """The only input to our grid - verse wrapper is the actual gverse domain itself

        Our class wraps the domain basically twice: it also wraps it into the
        Gridverse simulator, which provides functionality to convert abstract
        state and observations into numpy arrays.

        This is basically just an interface between the grid verse simulator
        and our code. Its biggest responsibility is translating the dictionary
        of numpy arrays generated by gridverse into a single array(see
        `flatten_state_or_observation` and `reshape_state`).

        Args:
            encoding_descr(`str`): compact, one-hot-orientation or one-hot-state
            env_descr(`str`): gym_minigrid description
        """
        POBNRLogger.__init__(self)

        self.log(
            POBNRLogger.LogLevel.V1,
            f"Initiating {env_descr} GridverseDomain with {encoding_descr} encoding",
        )

        gverse_env = gym_minigrid_from_descr(env_descr)

        self._gverse_obs_rep = DefaultObservationRepresentation(
            gverse_env.observation_space
        )

        self._gverse_env = gverse_env

        self._state_encoding = StateEncoding.construct(
            encoding_descr, DefaultStateRepresentation(gverse_env.state_space)
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

    @property
    def state(self) -> np.ndarray:
        """returns current state as numpy array"""
        return self._state_encoding.encode(self._gverse_env.state)

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

        - 1 if:
            - agent is on a wall or moving obstacle in new state
            - agent 'moved' but stayed on the same location
        1 if:
            - agent is on a goal in new state
        else 0
        """

        # pylint: disable=no-member

        grid, (y, x), _ = self._state_encoding.decode(new_state)
        item_under_agent = grid[y, x]

        if item_under_agent == Goal.type_index:
            return 1.0

        if item_under_agent in [MovingObstacle.type_index, Wall.type_index]:
            return -1

        grid, (prev_y, prev_x), _ = self._state_encoding.decode(state)

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

        grid, (y, x), _ = self._state_encoding.decode(new_state)
        item_under_agent = grid[y, x]

        grid, (prev_y, prev_x), _ = self._state_encoding.decode(state)

        # pylint: disable=no-member

        if item_under_agent in [
            Goal.type_index,
            Wall.type_index,
            MovingObstacle.type_index,
        ]:
            return True

        same_position = y == prev_y and x == prev_x
        translating = action in TRANSLATION_ACTIONS

        if translating and same_position:
            return True

        return False

    def sample_transition(
        self,
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """Samples a random transition in the domain

        First samples a state and action:

        - the state from the 'reset' function(which we assume generates all
          reachable states!)
            - explicitly set the position of the agent randomly
        - the action is sampled randomly from _our_ action space

        Then the next state and observation is sampled from the actual
        dynamics.

        Returns: `Tuple[np.ndarray, int, np.ndarray, np.ndarray]`: state - action - state - observation
        """

        a = GverseAction(self.action_space.sample())
        s = self._gverse_env.functional_reset()

        # set random agent, it is assumed here that (apart from the agent
        # position and orientation) all other reachable states are sampled
        # through the functional reset. Mind the walls
        s.agent.position = GversePosition(
            random.randint(1, self.h - 2), random.randint(1, self.w - 2)
        )
        s.agent.orientation = np.random.choice(Orientation)

        next_s, _, _ = self._gverse_env.functional_step(s, a)
        o = self._gverse_env.functional_observation(next_s)

        return (
            self._state_encoding.encode(s),
            a.value,
            self._state_encoding.encode(next_s),
            self._convert_gverse_obs(o),
        )

    def __repr__(self):
        return (
            f"Gridverse domain with with {self.action_space}, "
            f"observation space {self.observation_space} and "
            f"state space {self.state_space}"
        )

    def create_dynamics_observation_model(self) -> DynamicsModel.O:
        """returns the observation model of the domain

        The observation model is a s,a,s' -> p(o) model. This is a member
        function since the (numpy array) representation of these elements are
        kinda dependent on the implementation.

        Returns:
            `DynamicsModel.O`:
        """
        return ObservationModel(
            self.obs_h,
            self._state_encoding,
            self._gverse_env.state_space.max_grid_object_type,
        )

    def state_to_string(self, state: np.ndarray) -> str:
        _, pos, orient = self._state_encoding.decode(state)
        return f"Agent on {pos} facing {orient}"

    def observation_to_string(self, observation: np.ndarray) -> str:
        return "___"

    def action_to_string(self, action: int) -> str:
        return str(GverseAction(action))


def default_rollout_policy(
    state: np.ndarray, encoding: StateEncoding
) -> int:  # pylint: disable=unused-argument
    """rollout policy for Gridverse domain

    Basically samples 'forward' 80%, and otherwise turns either direction. The
    idea being is that this results in a random walk that covers relatively
    large amount of distance. Note that the policy, additionally, will not walk
    into walls.

    Follows the `POUCT.RolloutPolicy` 'interface'

    Args:
        state (`np.ndarray`): unused state
        encoding (`StateEncoding`): how the state is encoded

    Returns:
        int: sampled action
    """
    if random.random() < 0.8:
        grid, (y, x), orient = encoding.decode(state)

        # position in front of agent
        y, x = (
            np.array([y, x], dtype=int)
            + {
                Orientation.N: [-1, 0],
                Orientation.S: [1, 0],
                Orientation.E: [0, 1],
                Orientation.W: [0, -1],
            }[orient]
        )

        if min(x, y) >= 0 and max(x, y) < encoding.grid_size:

            # we safely can do this operation:
            cell_in_front = grid[y, x]

            if cell_in_front != Wall.type_index:  # pylint: disable=no-member
                return GverseAction.MOVE_FORWARD.value

    if random.choice([True, False]):
        return GverseAction.TURN_LEFT.value

    return GverseAction.TURN_RIGHT.value


def straight_or_turn_policy(
    state: np.ndarray, encoding: StateEncoding
) -> int:  # pylint: disable=unused-argument
    """rollout policy for Gridverse domain

    Goes straight forward unless faced with a wall, after which it will turn
    either left or right with 50% chance

    Follows the `POUCT.RolloutPolicy` 'interface'

    Args:
        state (`np.ndarray`): unused state
        encoding (`StateEncoding`): how the state is encoded

    Returns:
        int: sampled action
    """

    grid, (y, x), orient = encoding.decode(state)

    # position in front of agent
    y, x = (
        np.array([y, x], dtype=int)
        + {
            Orientation.N: [-1, 0],
            Orientation.S: [1, 0],
            Orientation.E: [0, 1],
            Orientation.W: [0, -1],
        }[orient]
    )

    if min(x, y) >= 0 and max(x, y) < encoding.grid_size:

        # we safely can do this operation:
        cell_in_front = grid[y, x]

        if cell_in_front != Wall.type_index:  # pylint: disable=no-member
            return GverseAction.MOVE_FORWARD.value

    if random.choice([True, False]):
        return GverseAction.TURN_LEFT.value

    return GverseAction.TURN_RIGHT.value
