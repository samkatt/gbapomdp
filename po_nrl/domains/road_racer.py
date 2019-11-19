""" the road racer domain """

import numpy as np

from po_nrl.environments import Environment, Simulator, ActionSpace
from po_nrl.environments import EnvironmentInteraction, SimulationResult
from po_nrl.misc import DiscreteSpace, Space, POBNRLogger


class RoadRacer(Environment, Simulator):
    """ represents the domain `RoadRacer`

    In this domain the agent is a driver on a highway. Its task is to navigate
    to the lanes to maximize the distance to the next car. The number of lanes,
    their length, and the average speed of other cars is variable.

    The agent can choose to either stay in lane, move up, or move down. The
    observation and reward is the distance until the next car on the current
    lane. If the agent attempts to move into another car, or off the road, then
    it is penalized.

    """

    GO_UP = 0
    NO_OP = 1
    GO_DOWN = 2

    def __init__(self, lane_length: int, lane_probs: np.ndarray):
        """ Creates a domain with lanes of length `lane_length` with transition probability `lane_probs`

        Args:
             lane_length: (`int`):
             lane_probs: (`np.ndarray`):

        """

        assert len(lane_probs) % 2 == 1, 'assume odd number of lanes'
        assert (lane_probs > 0).all() and (lane_probs <= 1).all(), 'expect 0 > probs > 1'
        assert lane_length > 2

        self.lane_length = lane_length
        self._lane_probs = lane_probs

        self.state = np.zeros(0)  # dummy declaraction
        self.reset()  # actually set state

        self.observations = DiscreteSpace([self.lane_length])
        self.actions = ActionSpace(3)

        self.states = DiscreteSpace([self.lane_length] * (self.num_lanes) + [self.num_lanes])

        self.logger = POBNRLogger('road racer')

    @property
    def lane_probs(self):
        """ returns the probabilities of each lane advancing """
        return self._lane_probs

    @lane_probs.setter
    def lane_probs(self, lane_probs):
        assert np.all(lane_probs >= .0) and np.all(lane_probs <= 1)

        self._lane_probs = lane_probs

    @property
    def num_lanes(self) -> int:
        """ returns the number of lanes

        Args:

        RETURNS (`int`):

        """
        return len(self.lane_probs)

    @property
    def current_lane(self) -> int:
        """ return the lane that the agent is currently in

        Args:

        RETURNS (`int`):

        """
        return RoadRacer.get_current_lane(self.state)

    @property
    def middle_lane(self) -> int:
        """ returns the middle lane

        Args:

        RETURNS (`int`):

        """
        return int(len(self.lane_probs) / 2)

    @property
    def agent_state_feature_index(self) -> int:
        """ returns the index of the state feature associated with the agent """
        return self.num_lanes

    @staticmethod
    def get_current_lane(state: np.ndarray) -> int:
        """ returns the lane in which the car is currently in

        Args:
             state: (`np.ndarray`):

        RETURNS (`int`):

        """
        assert state[-1] >= 0
        return state[-1]

    @staticmethod
    def get_observation(state: np.ndarray) -> np.ndarray:
        """ returns the (deterministic) observation associated with the state

        Args:
             state: (`np.ndarray`):

        RETURNS (`np.ndarray`):

        """
        return np.array([state[RoadRacer.get_current_lane(state)]])

    def reset(self) -> np.ndarray:
        """ implements `po_nrl.environments.Environment` """
        self.state = self.sample_start_state()

        return RoadRacer.get_observation(self.state)

    def step(self, action: int) -> EnvironmentInteraction:
        """ implements `po_nrl.environments.Environment.step`

        action: 0 -> 'go up', 1 -> 'stay', 2 -> 'go down'

        """

        step = self.simulation_step(self.state, action)
        reward = self.reward(self.state, action, step.state)
        terminal = self.terminal(self.state, action, step.state)

        self.state = step.state

        if self.logger.log_is_on(POBNRLogger.LogLevel.V2):
            self.logger.log(
                POBNRLogger.LogLevel.V2,
                f"lane={self.current_lane}, a={action}, o={step.observation[0]}, r={reward}"
            )

        return EnvironmentInteraction(step.observation, reward, terminal)

    @property
    def action_space(self) -> ActionSpace:
        """ implements `po_nrl.environments.Environment.action_space` """
        return self.actions

    @property
    def observation_space(self) -> Space:
        """ implements `po_nrl.environments.Environment.observation_space` """
        return self.observations

    @property
    def state_space(self) -> Space:
        """ implements `po_nrl.environments.Simulator.state_space` """
        return self.states

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """ implements `po_nrl.environments.Simulator.simulation_step`

        action: 0 -> 'go up', 1 -> 'stay', 2 -> 'go down'

        BUMPING rules:
            - actions that will go beyond the lanes have no effect on the lane
            - we first advance lanes, then try to move:
                * if a car is on the intended moved lane, agent stays put
            - legal car position **includes** 0

        """

        assert self.state_space.contains(state)
        assert self.action_space.contains(action)

        cur_lane = RoadRacer.get_current_lane(state)

        # advance lanes
        lane_advances = [np.random.rand() < p for p in self.lane_probs]

        if state[cur_lane] == 1:  # car in front of us does not move
            lane_advances[cur_lane] = 0

        # temporary next state: lanes have advanced, but
        # agent position has not been updated yet (hence [0])
        next_state = state - (lane_advances + [0])

        # move agent
        next_lane = min(max(0, cur_lane + action - 1), self.num_lanes - 1)

        if next_state[next_lane] != 0:
            next_state[self.agent_state_feature_index] = next_lane

        # cars re-appear at the start of the lane when finishing
        next_state[:-1] = next_state[:-1] % self.lane_length

        return SimulationResult(next_state, RoadRacer.get_observation(next_state))

    def sample_start_state(self) -> np.ndarray:
        """ implements `po_nrl.environments.Simulator.sample_start_state` """
        return np.concatenate((np.ones(self.num_lanes) * (self.lane_length - 1), [self.middle_lane])).astype(int)

    def obs2index(self, observation: np.ndarray) -> int:
        """ implements `po_nrl.environments.Simulator.obs2index` """
        assert self.observation_space.contains(observation)

        return observation[0]

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """ implements `po_nrl.environments.Simulator.reward`

        Reward consists of 2 components:

        1: the reward for being in the current lane
        2: the cost of potentially bumping

        The first component (1) is computed by taking the distance to the
        car in the current lane

        Bumping (2) happens when the agent either attempts to go off the lanes,
        or when it tries to go to a lane where a car is situated

        """

        assert self.state_space.contains(state)
        assert self.action_space.contains(action)
        assert self.state_space.contains(new_state)

        previous_lane = RoadRacer.get_current_lane(state)

        bump = (
            # action == 0 and previous_lane == 0 OR action == 2 and previous_lane == self.num_lanes - 1
            not action + previous_lane or action + previous_lane == self.num_lanes + 1\
            or\
            new_state[previous_lane - 1 + action] == 0  # destined spot has a car
        )

        # return the distance of the car in the current lane
        # penalize agent for moving (action of 0 or 2)
        return new_state[RoadRacer.get_current_lane(new_state)] - bump

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """ implements `po_nrl.environments.Simulator.terminal` """

        assert self.state_space.contains(state)
        assert self.action_space.contains(action)
        assert self.state_space.contains(new_state)

        return False  # continuous for now

    def __repr__(self) -> str:
        return f'Road racer of length {self.lane_length} with lane probabilities {self.lane_probs}'
