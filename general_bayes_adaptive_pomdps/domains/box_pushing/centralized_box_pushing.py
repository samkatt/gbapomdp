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

from general_bayes_adaptive_pomdps.domains.box_pushing.small_box_pushing import SmallBoxPushing
import one_to_one


class CentralizedBoxPushing(Domain):

    def __init__(
        self, grid_dim=(4, 4)
    ):
        """Construct the tiger domain

        Args:
             one_hot_encode_observation: (`bool`):
             correct_obs_probs: (`Optional[List[float]]`):

        """
        super().__init__()

        n_agents = 2

        self._env = SmallBoxPushing(grid_dim=grid_dim, n_agent=n_agents)

        grid_x, grid_y = grid_dim
        grid_ori = 4  # max 4 orientations per agent

        # state: (pos-x, pos-y, orientation) per agent, (pos-x, pos-y) per boxes (2 boxes)
        self._state_space = DiscreteSpace([grid_x, grid_y, grid_ori]*n_agents + [grid_x, grid_y]*2)

        # each agent has 4 cardinal movements
        self._action_space = ActionSpace(4**n_agents)

        # obs: the type of the pointed cell (empty, either box, wall, agent)
        self._obs_space = DiscreteSpace([4])

        self._state = self.sample_start_state()

        self.sem_action_space = one_to_one.JointNamedSpace(
            a1_a=one_to_one.RangeSpace(4),  # 0: move forward, 1: turn left, 2: turn right, 3: stay
            a2_a=one_to_one.RangeSpace(4),
        )
        self._action_lst = list(self.sem_action_space.elems)

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

    def sample_start_state(self) -> np.ndarray:
        """returns the (deterministic) start state

        Args:

        RETURNS (`np.ndarray`): [x, y, ori] per agent, [x, y] per box

        """
        self._env.reset()
        return self._env.get_state()

    def reset(self) -> np.ndarray:
        """reset the state

        Args:

        RETURNS (`np.ndarray`): [x, y, ori] per agent, [x, y] per box

        """
        self.sample_start_state()
        return self._env.get_obs()

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """simulates stepping from state using action. Returns interaction

        Args:
             state: (`np.ndarray`): [x, y, orientation] per agent, box positions
             action: (`int`):

        RETURNS (`general_bayes_adaptive_pomdps.core.SimulationResult`):

        """
        # reset to this state
        self._reset_sim(state)

        # convert action
        agent_actions = self._action_lst[action]

        # apply action
        self._env.step([agent_actions.a1_a.value, agent_actions.a2_a.value])

        # result of the action
        new_state = self._env.get_state()
        obs = self._env.get_obs()

        return SimulationResult(new_state, obs)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """total reward of 2 agents

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """
        # reset to this state
        self._reset_sim(state)

        # convert action
        agent_actions = self._action_lst[action]

        # apply action
        _, rewards, _, _ = self._env.step([agent_actions.a1_a.value, agent_actions.a2_a.value])

        # sum of all rewards
        return np.sum(rewards)

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if reached end in `new_state`

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`): whether the transition is terminal

        """
        # reset to this state
        self._reset_sim(state)

        # convert action
        agent_actions = self._action_lst[action]

        # apply action
        _, _, terminates, _ = self._env.step([agent_actions.a1_a.value, agent_actions.a2_a.value])

        # terminate when any box is pushed to the goal area
        return np.sum(terminates) > 0

    def step(self, action: int) -> DomainStepResult:
        """
        Combine all
        """
        # reset to this state
        self._reset_sim(self.state)

        # convert action
        agent_actions = self._action_lst[action]

        # apply action
        _, rewards, terminates, _ = self._env.step([agent_actions.a1_a.value, agent_actions.a2_a.value])

        # result of the action
        new_state = self._env.get_state()
        obs = self._env.get_obs()

        # sim_result = self.simulation_step(self.state, action)
        # reward = self.reward(self.state, action, sim_result.state)
        # terminal = self.terminal(self.state, action, sim_result.state)

        self.state = new_state

        return DomainStepResult(obs, np.sum(rewards), terminates)

    def _reset_sim(self, state):
        """
        Reset the simulator to a certain state
        """
        # 2 x (x, y, orientation) + 2 x (box-x, box-y)
        assert len(state) == 10
        self._env.reset(agents_pos=state[:6], boxes_pos=state[6:])
