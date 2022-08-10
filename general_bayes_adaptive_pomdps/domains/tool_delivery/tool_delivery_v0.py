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
    SimulationResult,
    TerminalState,
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
        # discrete locations: 2
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs
        # the correct tool order: [n_objs]*n_objs
        # human working step: 4
        n_objs = 3
        self.n_objs = n_objs
        self._state_space = DiscreteSpace([2] + [2]*n_objs +
                                          [2]*n_objs + [n_objs]*n_objs + [n_objs + 1] + [2])

        # OBSERVATION
        # discrete locations: 2
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs
        # human working step: 4
        self._obs_space = DiscreteSpace([2] + [2]*n_objs + [2]*n_objs + [n_objs + 1])

        # each agent has 4 possible actions: Get_Tool_i(0:3), Deliver
        self._action_space = ActionSpace(4)

    @property
    def state_space(self) -> DiscreteSpace:
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        return self._obs_space

    def get_state(self) -> np.ndarray:
        return self.core_env.get_state()

    def sample_start_state(self) -> np.ndarray:
        """returns the (deterministic) start state

        Args:

        RETURNS (`np.ndarray`):

        """
        self.core_env.reset()
        return self.core_env.get_state()

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
        # discrete locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs
        # the correct tool order: [n_objs]*n_objs
        # human working step: [4]
        # human currently accepts tool: [2]

        reward = -1

        prev_human_stage = state[-2]

        new_human_stage = new_state[-2]

        if prev_human_stage + 1 == new_human_stage:
            reward += 100

        correct_tool_orders = state[2*self.n_objs + 1 : 3*self.n_objs + 1]
        carrying_tools = state[1:self.n_objs + 1]

        # Deliver but not carrying a good tool
        if action == self.n_objs and prev_human_stage < self.n_objs:
            requested_tool_idx = correct_tool_orders[prev_human_stage]
            if carrying_tools[requested_tool_idx] != 1:
                # assert reward < 0, f"{state}, {requested_tool_idx}, {carrying_tools}"
                reward += -1

            # deliver but human is not accepting tools
            accept_tool = state[-1]
            if not accept_tool:
                reward += -10

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
        # discrete locations: [2]
        # which object in the basket: [2]*n_objs
        # which object are on the table: [2]*n_objs
        # the correct tool order: [n_objs]*n_objs
        # human working step: [4]
        # human currently accepts tool: [2]

        done = False
        human_stage = new_state[-1]

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
            perm = list(permutations([2, 1]))
            correct_tool_order = random.choice(perm)
        return ToolDeliveryV0([0] + list(correct_tool_order))

if __name__ == "__main__":
    env = ToolDeliveryV0(correct_tool_order=[0, 1, 2], render=False)
    env.reset()

    index_2_str = ['Get_Tool_0', 'Get_Tool_1', 'Get_Tool_2', 'Deliver']

    def action_to_str(action_idx):
        return f"Action: {index_2_str[action_idx]}"

    optimal = True

    rewards = []

    if optimal:

        act_list = [GET_TOOL_0] + [DELIVER] + [GET_TOOL_1] + [DELIVER] + [GET_TOOL_2] + [DELIVER]*3

        for action in act_list:
            print(action_to_str(action))
            state = env.step(action)
            print(state.observation, state.reward, state.terminal)
            print()
            rewards.append(state.reward)
            time.sleep(1)
    else:
        act_list = [GET_TOOL_0, GET_TOOL_1, GET_TOOL_2] + [DELIVER]*7

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
