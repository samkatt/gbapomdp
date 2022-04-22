"""The tiger problem implemented as domain"""

from logging import Logger
from typing import List, Optional

import numpy as np

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace, LogLevel
from general_bayes_adaptive_pomdps.models.tabular_bapomdp import DirCounts

from general_bayes_adaptive_pomdps.domains.small_box_pushing import SmallBoxPushing
import one_to_one


class MultiAgentBoxPushing(Domain):

    def __init__(
        self, grid_dim=(4, 4)
    ):
        """Construct the tiger domain

        Args:
             one_hot_encode_observation: (`bool`):
             correct_obs_probs: (`Optional[List[float]]`):

        """
        super().__init__()

        n_agent = 2

        self._env = SmallBoxPushing(grid_dim=grid_dim, n_agent=n_agent)

        self._state_space = DiscreteSpace([self._env.state_size])
        self._action_space = ActionSpace(self._env.action_size)
        self._obs_space = DiscreteSpace([self._env.obs_size])

        # For generating the true transition and observation tables
        self.sem_state_space = one_to_one.JointNamedSpace(
            a1_x=one_to_one.RangeSpace(grid_dim[0]),
            a1_y=one_to_one.RangeSpace(grid_dim[1]),
            a1_ori=one_to_one.RangeSpace(4),

            a2_x=one_to_one.RangeSpace(grid_dim[0]),
            a2_y=one_to_one.RangeSpace(grid_dim[1]),
            a2_ori=one_to_one.RangeSpace(4),

            b1_x=one_to_one.RangeSpace(grid_dim[0]),
            b1_y=one_to_one.RangeSpace(grid_dim[1]),

            b2_x=one_to_one.RangeSpace(grid_dim[0]),
            b2_y=one_to_one.RangeSpace(grid_dim[1]),
        )

        self._state_lst = list(self.sem_state_space.elems)

        self.sem_obs_space = one_to_one.JointNamedSpace(
            a1_fcell=one_to_one.RangeSpace(4),  # Front cell can be empty (0), box (1), wall (2), or agent (3)
            a2_fcell=one_to_one.RangeSpace(4),  # Front cell can be wall, agent, empty, or box
        )
        self._obs_lst = list(self.sem_obs_space.elems)

        self.sem_action_space = one_to_one.JointNamedSpace(
            a1_a=one_to_one.RangeSpace(4),  # 0: move forward, 1: turn left, 2: turn right, 3: stay
            a2_a=one_to_one.RangeSpace(4),
        )
        self._action_lst = list(self.sem_action_space.elems)

        self._state = self.sample_start_state()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        self._state = state

    @property
    def state_space(self) -> DiscreteSpace:
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        return self._obs_space

    def _convert_raw_state(self, state, raw_state):
        assert(len(raw_state) == 10)

        state.a1_x.value = raw_state[0]
        state.a1_y.value = raw_state[1]
        state.a1_ori.value = raw_state[2]

        state.a2_x.value = raw_state[3]
        state.a2_y.value = raw_state[4]
        state.a2_ori.value = raw_state[5]

        state.b1_x.value = raw_state[6]
        state.b1_y.value = raw_state[7]

        state.b2_x.value = raw_state[8]
        state.b2_y.value = raw_state[9]

        return state

    def _convert_raw_obs(self, obs, raw_obs):
        assert(len(raw_obs) == 2)
        obs.a1_fcell.value = raw_obs[0]
        obs.a2_fcell.value = raw_obs[1]
        return obs

    def sample_start_state(self) -> np.ndarray:
        _, raw_state = self._env.reset()

        # Convert raw_state to an indexed state
        state = self._state_lst[0]
        state = self._convert_raw_state(state, raw_state)

        return np.array([state.idx], dtype=int)

    def reset(self) -> np.ndarray:
        self._state = self.sample_start_state()
        raw_obs = self._env._getobs()
        obs = self._obs_lst[0]
        obs = self._convert_raw_obs(obs, raw_obs)

        return np.array([obs.idx])

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        self._reset_sim(state)

        # convert action
        agent_actions = self._action_lst[action]

        self._env.step([agent_actions.a1_a, agent_actions.a2_a])

        raw_state = self._env.get_state()
        raw_obs = self._env._getobs()

        state = self._state_lst[0]
        state = self._convert_raw_state(state, raw_state)

        obs = self._obs_lst[0]
        obs = self._convert_raw_obs(state, raw_obs)

        return SimulationResult(np.array([state.idx]), np.array([obs.idx]))

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        self._reset_sim(state)

        # convert action
        agent_actions = self._action_lst[action]

        _, rewards, _, _ = self._env.step([agent_actions.a1_a, agent_actions.a2_a])

        return np.sum(rewards)

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        self._reset_sim(state)

        # convert action
        agent_actions = self._action_lst[action]

        _, _, terminates, _ = self._env.step([agent_actions.a1_a, agent_actions.a2_a])

        return np.sum(terminates) > 0  # terminate when any box is pushed to the goal area

    def step(self, action: int) -> DomainStepResult:
        sim_result = self.simulation_step(self.state, action)
        reward = self.reward(self.state, action, sim_result.state)
        terminal = self.terminal(self.state, action, sim_result.state)

        self.state = sim_result.state

        return DomainStepResult(sim_result.observation, reward, terminal)

    def _reset_sim(self, state):
        state = self._state_lst[state[0]]
        a1_pos = [state.a1_x.value, state.a1_y.value, state.a1_ori.value]
        a2_pos = [state.a2_x.value, state.a2_y.value, state.a2_ori.value]
        b1_pos = [state.b1_x.value, state.b1_y.value]
        b2_pos = [state.b2_x.value, state.b2_y.value]

        self._env.reset(agent_pos=a1_pos + a2_pos, boxes_pos=b1_pos + b2_pos)
