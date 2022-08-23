"""The tiger problem implemented as domain"""

from ast import List
from typing import Tuple
from general_bayes_adaptive_pomdps.domains.warehouse.env_warehouse import TurtleBot

import numpy as np
import random
import time

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace

from general_bayes_adaptive_pomdps.domains.tool_delivery.single_human import ObjSearchDelivery_v4 as EnvToolDelivery

from itertools import permutations

GET_TOOL_0 = 0
GET_TOOL_1 = 1
GET_TOOL_2 = 2
DELIVER = 3

class ToolDeliveryV0(Domain):

    def __init__(self, correct_tool_order, render=False):
        """Construct domain
        Args:
        """
        super().__init__()

        self.correct_tool_order = correct_tool_order
        self.core_env = EnvToolDelivery(correct_tool_order, render=render)

        # STATE
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs
        # human working step: [n_objs + 1]
        # accept tool/not [2]
        n_objs = 3
        self.n_objs = n_objs
        self._state_space = DiscreteSpace([2] + [2]*n_objs +
                                          [2]*n_objs + [n_objs + 1] + [2])

        # OBSERVATION
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human working step: [n_objs + 1] (only observable in the work-room)
        self._obs_space = DiscreteSpace([2] + [2]*n_objs + [2]*n_objs + [n_objs + 1])
        self._oy_obs_space = DiscreteSpace([2]*n_objs + [n_objs + 1])

        # each agent has 4 possible actions: Get_Tool_i(0:n_objs - 1), Deliver
        self._action_space = ActionSpace(n_objs + 1)

    @property
    def state_space(self) -> DiscreteSpace:
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        return self._obs_space

    @property
    def oy_observation_space(self) -> DiscreteSpace:
        return self._oy_obs_space

    def get_state(self) -> np.ndarray:
        return self.core_env.get_state()

    def oy_from_o(self, o):
        """extract oy component from an observation"""
        # OBSERVATION
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human working step: [n_objs + 1] (only observable in the work-room)
        return o[1 + self.n_objs:]

    def sample_start_state(self) -> np.ndarray:
        """returns the (deterministic) start state

        Args:

        RETURNS (`np.ndarray`):

        """

        # STATE
        # discrete room locations: [2] - 0: tool-room
        # which object in the basket: [2]*n_objs - 0: not in the basket
        # which object are on the table: [2]*n_objs - 0: on the table
        # human working step: [n_objs + 1] - 0: start at 0
        # accept tool/not [2] - 0: not accept tools

        init_state = np.zeros(self._state_space.ndim)
        return np.array(init_state, dtype=int)

    def reset(self) -> np.ndarray:
        """reset the environment

        Args:

        RETURNS (`np.ndarray`)

        """
        return self.core_env.reset()

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """a known reward function

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """
        # STATE
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs
        # human working step: [n_objs + 1]
        # accept tool/not [2]

        reward = -1

        prev_human_stage = state[-2]

        new_human_stage = new_state[-2]

        # Deliver a good tool
        if prev_human_stage + 1 == new_human_stage and action == self.n_objs:
            reward += 100

        # Deliver but not carrying a good tool (stage does not change)
        if prev_human_stage == new_human_stage and action == self.n_objs:
            reward += -10

            # deliver but human is currently working and not accept tool
            accept_tool = state[-1]
            if not accept_tool:
                reward += -10

        # Penalize when carrying more than 1 tool
        carrying_tools = state[1:1 + self.n_objs]
        total_tools = np.sum(carrying_tools)
        penalty = max(0.0, total_tools - 1)
        reward += -30*penalty

        return reward

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if reached end in `new_state`

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`): whether the transition is terminal

        """
        # STATE
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs
        # human working step: [n_objs + 1]
        # accept tool/not [2]

        done = False
        human_stage = new_state[-2]

        if human_stage == 3:
            done = True

        return done

    def step(self, action: int) -> DomainStepResult:
        s = self.core_env.get_state()
        next_obs, _, _, _ = self.core_env.step([action])
        next_s = self.core_env.get_state()
        r = self.reward(s, action, next_s)
        t = self.terminal(s, action, next_s)
        return DomainStepResult(next_obs[0], r, t)

class ToolDeliveryV0Prior(DomainPrior):
    """perfect prior for now
    """

    def __init__(self):
        """initiate the prior

        Args:
        """
        super().__init__()

    def sample(self, prior=None) -> Domain:
        """
        Prior that the correct order will start with the first tool
        """
        if prior:
            correct_tool_order = list(prior)
        else:
            perm = list(permutations([0, 2, 1]))
            correct_tool_order = random.choice(perm)
        return ToolDeliveryV0(list(correct_tool_order))

if __name__ == "__main__":
    env = ToolDeliveryV0(correct_tool_order=[2, 1, 0], render=False)
    env.reset()

    index_2_str = ['Get_Tool_0', 'Get_Tool_1', 'Get_Tool_2', 'Deliver']

    def action_to_str(action_idx):
        return f"Action: {index_2_str[action_idx]}"

    optimal = False

    rewards = []

    if optimal:

        act_list = [GET_TOOL_2] + [DELIVER] + [GET_TOOL_1] + [DELIVER] + [GET_TOOL_0] + [DELIVER]

        for action in act_list:
            print(action_to_str(action))
            state = env.step(action)
            print(state.observation, state.reward, state.terminal)
            print()
            rewards.append(state.reward)
            time.sleep(1)
    else:
        act_list = [GET_TOOL_0, GET_TOOL_1, GET_TOOL_2] + [DELIVER]*3

        for action in act_list:
            print(action_to_str(action))
            state = env.step(action)
            print(state.reward, state.terminal)
            print()
            rewards.append(state.reward)
            time.sleep(1)

    print(rewards)
    discounted_return = sum(
        pow(0.95, i) * r for i, r in enumerate(rewards)
    )

    print(optimal, discounted_return)
