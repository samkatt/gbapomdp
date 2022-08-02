"""The tiger problem implemented as domain"""

from ast import List
from typing import Tuple
from general_bayes_adaptive_pomdps.domains.warehouse.env_warehouse import TurtleBot

import numpy as np
import random

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
    TerminalState,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace

from general_bayes_adaptive_pomdps.domains.warehouse.env_warehouse import EnvWareHouse, Human
import one_to_one

from itertools import permutations

Go_to_WR = 0
Go_to_TR = 1
Get_tool_1 = 2
Get_tool_2 = 3
Get_tool_3 = 4
Get_tool_4 = 5
Deliver_tool = 6
No_Op = 7

class WareHouseV0(Domain):

    def __init__(self, correct_tool_order):
        """Construct domain
        The agent is allowed to see the desired tool when at the work room
        The prior assumes the first desired tool is always the tool#1
        Args:
        """
        super().__init__()

        self.correct_tool_order = correct_tool_order

        # a state s = (x, y) and an observation = (x, z)
        # x is part of the state that is fully observable

        # location: 0 (tool-room) or 1 (work-room) (2)
        # current-tool carried by the turtlebot: 0, 1, 2, 3, 4 (5)
        # task-stage: 0, 1, 2, 3, 4 (5)
        # human step count: 0, 1, 2 (3)
        # human status: 0, 1 (2)
        # human-desired-tool: 0, 1, 2, 3 (4) 0: means tool#1

        # requirements: desired tool = correct_tool_order[task-stage]
        # human status = 0 then human-step-cnt = 0

        self._state_space = DiscreteSpace([2, 5, 5, 3, 2, 4])

        # obs: (location, current-tool) per turtlebot + (human-status, desired-tool-index)

        # location: tool-room or work-room (2)
        # current-tool: 0, 1, 2, 3, 4 (5)
        # human-status: working, waiting (2) (only available when one turtlebot is at the work room)
        # task-stage: 0, 1, 2, 3, 4 (5) (only available when at the work room)
        # or desired-tool: 0, 1, 2, 3 (4) (only available when at the work room)
        self._obs_space = DiscreteSpace([2, 5, 2, 4])

        self._state = self.sample_start_state()

        # each agent has 8 possible actions: Go_to_WR, Go_to_TR, Get_Tool_i (i=1, 2, 3, 4), Deliver_Tool, No_Op
        self._action_space = ActionSpace(8)

        self.sem_action_space = one_to_one.JointNamedSpace(
            a1=one_to_one.RangeSpace(8),
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

    def get_xstate(self) -> np.ndarray:
        return self._state[:2]

    def sample_start_state(self) -> np.ndarray:
        """returns the (deterministic) start state

        Args:

        RETURNS (`np.ndarray`):

        """
        env = EnvWareHouse(self.correct_tool_order)
        env.reset()
        return env.get_state()

    def sample_start_ystate(self) -> np.ndarray:
        """returns the (deterministic) start y component of state

        Args:

        RETURNS (`np.ndarray`):

        """
        env = EnvWareHouse(self.correct_tool_order)
        env.reset()
        return env.get_state()[2:]

    def generate_observation(self, state: np.ndarray = None) -> np.ndarray:
        """generates an observation of the state

        Args:
             state: (`np.ndarray`): 

        RETURNS (`np.ndarray`):

        """
        if state is None:
            state = self.state

        # obs: (location, current-tool) per turtlebot + (human-status, desired-tool)
        # location: tool-room or work-room (2)
        # current-tool: 0, 1, 2, 3, 4 (5)

        # these only useful when the turtlebot is in the workroom
        # human-status: working, waiting, null (3)
        # task-stage: 0, 1, 2, 3, 4 (5)

        # NOTE: start with observing the desired tool for now

        # location: 0 (tool-room) or 1 (work-room) (2)
        # current-tool carried by the turtlebot: 0, 1, 2, 3, 4 (5)
        # task-stage: 0, 1, 2, 3, 4 (5)
        # human step count: 0, 1, 2 (3)
        # human status: 0, 1 (2)
        # human-desired-tool: 0, 1, 2, 3 (4) 0: means tool#1


        bot_location = state[0]
        if bot_location == TurtleBot.AT_WORK_ROOM:
            task_stage = state[2]
            human_status = state[4]
            desired_tool = state[5]
        else:
            human_status = 0
            task_stage = 0
            desired_tool = 0

        obs = [state[0], state[1], human_status, desired_tool]

        return np.array(obs, dtype=int)

    def extract_x(self, o):
        """extract the x component of an observation"""
        return o[:2]

    def reset(self) -> np.ndarray:
        """reset the state

        Args:

        RETURNS (`np.ndarray`): [x, y, ori] per agent, [x, y] per box

        """
        self._state = self.sample_start_state()
        return self.generate_observation()

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """simulates stepping from state using action. Returns interaction

        Args:
             state: (`np.ndarray`):
             action: (`int`):

        RETURNS (`general_bayes_adaptive_pomdps.core.SimulationResult`):

        """

        assert self.action_space.contains(action), f"action {action} not in space"
        assert self.state_space.contains(state), f"state {state} not in space"

        env = EnvWareHouse(self.correct_tool_order)

        # reset to this state
        env.reset(state)

        # convert action
        agent_actions = self._action_lst[action]

        # apply actions
        env.step([agent_actions.a1.value])

        # result of the action
        new_state = env.get_state()
        obs = self.generate_observation(new_state)

        return SimulationResult(new_state, obs)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """total reward of 2 agents

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """

        reward = -1

        prev_task_stage = state[2]
        current_task_stage = new_state[2]

        # human just receives a desired tool
        if prev_task_stage != current_task_stage and action == Deliver_tool:
            reward += 10

        return reward

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if reached end in `new_state`

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`): whether the transition is terminal

        """
        done = False
        # location: 0 (tool-room) or 1 (work-room) (2)
        # current-tool carried by the turtlebot: 0, 1, 2, 3, 4 (5)
        # task-stage: 0, 1, 2, 3, 4 (5)
        # human step count: 0, 1, 2 (3)
        # human status: 0, 1 (2)
        # human-desired-tool: 0, 1, 2, 3 (4) 0: means tool#1
        # the current desired tool is the final tool and human just receives it
        current_human_status = new_state[4]
        current_desired_tool = new_state[5]
        task_stage = new_state[2]
        if current_desired_tool + 1 == self.correct_tool_order[-1] \
            and current_human_status == Human.WORKING \
            and task_stage == len(self.correct_tool_order):
            done = True

        return done

    def isValidState(self, s: np.ndarray) -> bool:
        # current-tool carried by the turtlebot: 0, 1, 2, 3, 4 (5)
        # task-stage: 0, 1, 2, 3, 4 (5)
        # human step count: 0, 1, 2 (3)
        # human status: 0, 1 (2)
        # human-desired-tool: 0, 1, 2, 3 (4) 0: means tool#1
        # the current desired tool is the final tool and human just receives it

        carrying_tool = s[1]
        (task_stage, human_step_cnt, human_status, human_desired_tool) = s[2:6]
        assert 0 <= task_stage <= 4
        assert 0 <= human_step_cnt <= 2
        assert 0 <= human_status <= 1
        assert 0 <= human_desired_tool <= 3

        # check if the state is a terminal one
        if self.terminal(None, None, s):
            return False

        # check if the desired tool is compatible to the task_stage
        if task_stage == 4:
            if human_desired_tool != self.correct_tool_order[-1] - 1:
                return False
        else:
            if human_desired_tool != self.correct_tool_order[task_stage] - 1:
                return False

        # If carrying tool in the desired list, it shouldn't be delivered before
        if carrying_tool != 0 and carrying_tool in self.correct_tool_order:
            index = self.correct_tool_order.index(carrying_tool)

            if task_stage > index:
                return False

        if human_status == Human.WAITING_TOOL:
            # if human is waiting, step count should be zero
            if human_step_cnt != 0:
                return False
        else:
            # if not, step count should be non-zero
            if human_step_cnt == 0:
                return False

        return True

    def step(self, act: int) -> DomainStepResult:
        """
        Combine all
        """
        env = EnvWareHouse(self.correct_tool_order)

        # reset to this state
        env.reset(self.state)

        sim_step = self.simulation_step(self.state, act)
        reward = self.reward(self.state, act, sim_step.state)
        terminal = self.terminal(self.state, act, sim_step.state)

        self.state = sim_step.state

        return DomainStepResult(sim_step.observation, reward, terminal)

class WareHouseV0Prior(DomainPrior):
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
            perm = list(permutations([2, 3, 4]))
            correct_tool_order = random.choice(perm)
        return WareHouseV0([1] + list(correct_tool_order))

if __name__ == "__main__":
    env = WareHouseV0(correct_tool_order=[1, 4, 3, 2])
    env.reset()

    ACT_LST = ['Go_to_WR', 'Go_to_TR', 'Get_Tool_0', 'Get_Tool_1',
               'Get_Tool_2', 'Get_Tool_3', 'Deliver', 'No_Op']


    def action_to_str(action_idx):
        return f"Action: {ACT_LST[action_idx]}"

    ACT_LST = [Go_to_TR, Get_tool_1, Go_to_WR, Get_tool_1, Get_tool_2, No_Op, Get_tool_4, Go_to_WR]
    ACT_LST += [Get_tool_2, Get_tool_1, Get_tool_1, Deliver_tool]

    # action_list = [Deliver_tool, Go_to_TR, Get_tool_1, Go_to_WR, Deliver_tool, Deliver_tool, Deliver_tool, Get_tool_3]
    # action_list += [Go_to_TR, Get_tool_2, Go_to_WR, Deliver_tool, Go_to_TR, Get_tool_3, Go_to_WR, Deliver_tool]
    # action_list += [Go_to_TR, Get_tool_4, Go_to_WR, Deliver_tool]

    for action in ACT_LST:
        print(action_to_str(action))
        state = env.step(action)
        print(env.state)
        print(state.reward)
        print(env.generate_observation(env.state))
        print()

    print(env.terminal(0, 0, env.state))
