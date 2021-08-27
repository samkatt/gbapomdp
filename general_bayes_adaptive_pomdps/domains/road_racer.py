""" the road racer domain """
from logging import Logger

import numpy as np

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
    TerminalState,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace, LogLevel


class RoadRacer(Domain):
    """represents the domain `RoadRacer`

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
    LANE_LENGTH = 6

    def __init__(self, lane_probs: np.ndarray):
        """Creates a domain with transition probability `lane_probs`

        Args:
             lane_probs: (`np.ndarray`):

        """
        super().__init__()

        assert len(lane_probs) % 2 == 1, f"assume odd number of lanes, not {lane_probs}"
        assert (lane_probs > 0).all() and (
            lane_probs <= 1
        ).all(), f"expect 0 > probs > 1, not {lane_probs}"

        self._lane_probs = lane_probs

        self.state = np.zeros(0)  # dummy declaraction
        self.reset()  # actually set state

        self.observations = DiscreteSpace([self.lane_length])
        self.actions = ActionSpace(3)

        self.states = DiscreteSpace(
            [self.lane_length] * (self.num_lanes) + [self.num_lanes]
        )

        self.logger = Logger(self.__class__.__name__)

    @property
    def lane_length(self) -> int:
        """returns the length of the lanes"""
        return self.LANE_LENGTH

    @property
    def lane_probs(self):
        """returns the probabilities of each lane advancing"""
        return self._lane_probs

    @lane_probs.setter
    def lane_probs(self, lane_probs):
        assert np.all(lane_probs >= 0.0) and np.all(lane_probs <= 1)

        self._lane_probs = lane_probs

    @property
    def num_lanes(self) -> int:
        """returns the number of lanes

        Args:

        RETURNS (`int`):

        """
        return len(self.lane_probs)

    @property
    def current_lane(self) -> int:
        """return the lane that the agent is currently in

        Args:

        RETURNS (`int`):

        """
        return RoadRacer.get_current_lane(self.state)

    @property
    def middle_lane(self) -> int:
        """returns the middle lane

        Args:

        RETURNS (`int`):

        """
        return int(len(self.lane_probs) / 2)

    @property
    def agent_state_feature_index(self) -> int:
        """returns the index of the state feature associated with the agent"""
        return self.num_lanes

    @staticmethod
    def get_current_lane(state: np.ndarray) -> int:
        """returns the lane in which the car is currently in

        Args:
             state: (`np.ndarray`):

        RETURNS (`int`):

        """
        assert state[-1] >= 0
        return state[-1]

    @staticmethod
    def get_observation(state: np.ndarray) -> np.ndarray:
        """returns the (deterministic) observation associated with the state

        Args:
             state: (`np.ndarray`):

        RETURNS (`np.ndarray`):

        """
        return np.array([state[RoadRacer.get_current_lane(state)]])

    def reset(self) -> np.ndarray:
        """implements `general_bayes_adaptive_pomdps.core.Environment`"""
        self.state = self.sample_start_state()

        return RoadRacer.get_observation(self.state)

    def step(self, action: int) -> DomainStepResult:
        """implements `general_bayes_adaptive_pomdps.core.Environment.step`

        action: 0 -> 'go up', 1 -> 'stay', 2 -> 'go down'

        """

        step = self.simulation_step(self.state, action)
        reward = self.reward(self.state, action, step.state)
        terminal = self.terminal(self.state, action, step.state)

        self.state = step.state

        self.logger.log(
            LogLevel.V2.value,
            "lane=%d and lanes %s, a=%d, o=%d, r=%f",
            self.current_lane,
            self.state[:-1],
            action,
            step.observation[0],
            reward,
        )

        return DomainStepResult(step.observation, reward, terminal)

    @property
    def action_space(self) -> ActionSpace:
        """Part of :class:`Domain` protocol"""
        return self.actions

    @property
    def observation_space(self) -> DiscreteSpace:
        """implements `general_bayes_adaptive_pomdps.environments.Environment.observation_space`"""
        return self.observations

    @property
    def state_space(self) -> DiscreteSpace:
        """Part of :class:`Domain` protocol"""
        return self.states

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """Part of :class:`Domain` protocol

        action: 0 -> 'go up', 1 -> 'stay', 2 -> 'go down'

        BUMPING rules:
            - actions that will go beyond the lanes have no effect on the lane
            - we first advance lanes, then try to move:
                * if a car is on the intended moved lane, agent stays put
            - legal car position **includes** 0

        """

        assert self.state_space.contains(state)
        assert self.action_space.contains(action)

        if state[RoadRacer.get_current_lane(state)] == 0:
            raise TerminalState(f"state {state} is terminal because agent on car")

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
        """Part of :class:`Domain` protocol"""
        return np.concatenate(
            (np.ones(self.num_lanes) * (self.lane_length - 1), [self.middle_lane])
        ).astype(int)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """Part of :class:`Domain` protocol

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

        desired_lane = RoadRacer.get_current_lane(state) - 1 + action
        current_lane = RoadRacer.get_current_lane(new_state)

        bump = action != RoadRacer.NO_OP and desired_lane != current_lane

        # return the distance of the car in the current lane
        # penalize agent for bumping
        return new_state[current_lane] - bump

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """Part of :class:`Domain` protocol"""

        assert self.state_space.contains(state), str(state)
        assert self.action_space.contains(action), str(action)
        assert self.state_space.contains(new_state), str(new_state)

        return False  # continuous for now

    def __repr__(self) -> str:
        return f"Road racer of length {self.lane_length} with lane probabilities {self.lane_probs}"


class RoadRacerPrior(DomainPrior):
    """standard prior over the road racer domain

    The agent's transition and observation model is known, however the other
    cars' speed is not. We assign a expected model of p=.5 for advancing to
    each lane.

    """

    def __init__(self, num_lanes: int, num_total_counts: float):
        """initiate the prior, will make observation one-hot encoded

        Args:
             num_lanes: (`float`): number of lanes in the domain
             num_total_counts: (`float`): number of total counts of Dir prior

        """
        super().__init__()

        if num_total_counts <= 0:
            raise ValueError("Assume positive number of total counts")

        self._total_counts = num_total_counts
        self._num_lanes = num_lanes

    def sample(self) -> Domain:
        """returns a Road Racer instance with some sampled set of lane speeds

        The prior over each lane advancement probability  is .5

        RETURNS (`general_bayes_adaptive_pomdps.domain.Domain`):

        """
        sampled_lane_speeds = np.random.beta(
            0.5 * self._total_counts, 0.5 * self._total_counts, self._num_lanes
        )

        return RoadRacer(sampled_lane_speeds)
