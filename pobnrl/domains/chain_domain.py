""" chain domain environment """

import numpy as np

from environments import Environment, EnvironmentInteraction, ActionSpace
from environments import Simulator, SimulationResult
from misc import DiscreteSpace, POBNRLogger


class ChainDomain(Environment, Simulator, POBNRLogger):  # pylint: disable=too-many-instance-attributes
    """ the chain environment

    The domains are indexed by problem size N and action mask W =
    Ber(0.5)^NxN,with S={0,1}^NxN and A={0,1}. he agent begins each episode
    inthe upper left-most state in the grid and deterministically falls one row
    per time step. The state encodes the agent’s row and column as a one-hot
    vectorst. The actions {0,1} move the agent left or right depending on the
    action mask W at state s, which remains fixed. The agent incurs a cost of
    0.01/N for moving right in all states except for the right-most, in which
    the reward is 1. The reward for action left is always zero. An episode ends
    after N time steps so that the optimal policy is to move right each step
    and receive a total return of 0.99; all other policies receive zero or
    negative return

    """

    def __init__(self, size: int, one_hot_observations):
        """ construct the chain environment

        Args:
             size: (`int`): size of the grid

        """

        assert size > 1, "Please enter domain size > 1"

        POBNRLogger.__init__(self)

        self._size = size
        self._move_cost = .01 / self.size
        self._one_hot_observations = one_hot_observations

        self._state_space = DiscreteSpace([self.size, self.size])
        self._action_space = ActionSpace(2)

        if self._one_hot_observations:
            self._observation_space = DiscreteSpace([2] * self.size * self.size)
        else:
            self._observation_space = self.state_space  # (x,y)

        self._action_mapping = np.random.binomial(1, .5, self.size)
        self._state = self.sample_start_state()

    @property
    def state(self) -> np.ndarray:
        """ returns current state

        Args:

        RETURNS (`np.ndarray`): [x,y] positions of agent

        """
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """ sets state

        Args:
             state: (`np.ndarray`): [x,y] agent position

        """
        assert self.size > state[0] >= 0
        assert self.size > state[1] > 0
        assert state[0] <= (self.size - state[1])

        self._state = state

    @property
    def state_space(self) -> DiscreteSpace:
        """ `pobnrl.misc.DiscreteSpace`([size,size])"""
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        """ a `pobnrl.environments.ActionSpace` space with 2 actions"""
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `pobnrl.misc.DiscreteSpace`, depends on `one_hot_observation` flag for `this`"""
        return self._observation_space

    def sample_start_state(self) -> np.ndarray:
        """ samples the (deterministic) start state

        Args:

        RETURNS (`np.ndarray`): [x,y]

        """

        # x, level (size-1...0)
        return np.array([0, self.size - 1])

    @property
    def size(self):
        """ returns size (length) of (square) grid """
        return self._size

    def state2observation(self, state: np.ndarray = None):
        """ returns a 1-hot encoding of the state

        Will use self.state if provided state is not given

        Args:
             state: (`np.ndarray`): the state [x,y]

        """

        # default value of state
        if state is None:
            state = self.state

        if not self._one_hot_observations:
            return state

        obs = np.zeros((self.size, self.size))
        obs[state[0], state[1]] = 1

        return obs.reshape(self.size * self.size)

    def reset(self):
        """ resets internal state and return first observation """

        self._state = self.sample_start_state()
        return self.state2observation()

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """ updates the state depending on action

        Depending on the state, action can either move the agent left or rigth
        deterministically. The observation is a noise-less 1-hot-encoding of
        the state and the environment stops when the end is reached (on either
        x-axis)

        Args:
             state: (`np.ndarray`): the state [x,y]
             action: (`int`): 0 is action A, 1 = action B

        RETURNS (`pobnrl.environments.SimulationResult`): the transition

        """

        assert 2 >= action >= 0, "expecting action A or B"

        new_state = state.copy()

        # move horizontally
        if action == self._action_mapping[state[0]]:
            new_state[0] += 1
        else:
            new_state[0] = max(0, state[0] - 1)  # bound on grid

        # move vertically
        new_state[1] -= 1

        return SimulationResult(new_state, self.state2observation(new_state))

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """ reward for finding end, otherwise slight penalty for going to end

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """

        assert self.state_space.contains(state)
        assert self.state_space.contains(new_state)
        assert self.action_space.contains(action)

        reward = 0 if action != self._action_mapping[state[0]] else -self._move_cost

        if new_state[0] == self.size - 1:
            reward += 1

        return reward

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """ the termination function: agent reaches bottom

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`): whether the transition is terminal

        """

        assert self.state_space.contains(state)
        assert self.state_space.contains(new_state)
        assert self.action_space.contains(action)

        return new_state[1] == 0

    def step(self, action: int) -> EnvironmentInteraction:
        """ performs a step in the domain depending on action

        Depending on the state, action can either move the agent left or rigth
        deterministically. The observation is a noise-less 1-hot-encoding of
        the state and the environment stops when the end is reached (on either
        x-axis)

        Args:
             action: (`int`): 0 is action A, 1 = action B

        RETURNS (`pobnrl.environments.EnvironmentInteraction`): the transition

        """

        sim_step = self.simulation_step(self.state, action)

        reward = self.reward(self.state, action, sim_step.state)
        terminal = self.terminal(self.state, action, sim_step.state)

        if self.log_is_on(POBNRLogger.LogLevel.V2):
            self.log(
                POBNRLogger.LogLevel.V2,
                f"Agent moved {self.state} -> {sim_step.state}"
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
            f"{observation} not in observation space {self.observation_space}"
        assert np.all(self.size > observation) and np.all(observation >= 0)

        if not self._one_hot_observations:
            return self.observation_space.index_of(observation)

        # one-hot

        assert np.sum(observation) == 1
        return observation.argmax()
