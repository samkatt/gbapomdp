"""The tiger problem implemented as domain"""

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

from general_bayes_adaptive_pomdps.domains.warehouse_v0.env_warehouse import EnvWareHouse, Human
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

class WareHouse(Domain):

    def __init__(self, correct_tool_order=None):
        """Construct the tiger domain
        Args:
            grid_dim

        """
        super().__init__()

        self.n_turtlebots = 1

        if correct_tool_order is None:
            self.correct_tool_order = [1, 2, 3, 4]
        else:
            self.correct_tool_order = correct_tool_order

        # state: (location, current-tool) per turtlebot + (human-status, desired-tool, human-step-cnt)

        # location: 0 (tool-room) or 1 (work-room) (2)
        # current-tool: 0, 1, 2, 3, 4 (5)
        # human-status: working or waiting (2)
        # desired-tool: 0, 1, 2, 3, 4 (5) 0: means tool#1
        # human step count: 0, 1, 2 (3)
        self._state_space = DiscreteSpace([2, 5]*self.n_turtlebots + [2, 5] + [3])

        # obs: (location, current-tool) per turtlebot + (human-status, desired-tool-index)

        # location: tool-room or work-room (2)
        # current-tool: 0, 1, 2, 3, 4 (5)
        # human-status: working, waiting (2) (only available when one turtlebot is at the work room)
        # task-stage: 0, 1, 2, 3, 4 (5) (only available when one turtlebot is at the work room)
        self._obs_space = DiscreteSpace([2, 5]*self.n_turtlebots + [2, 5])

        self._state = self.sample_start_state()

        # each agent has 8 possible actions: Go_to_WR, Go_to_TR, Get_Tool_i (i=1, 2, 3, 4), Deliver_Tool, No_Op
        self._action_space = ActionSpace(8**self.n_turtlebots)

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

    def sample_start_state(self) -> np.ndarray:
        """returns the (deterministic) start state

        Args:

        RETURNS (`np.ndarray`):

        """
        env = EnvWareHouse(self.correct_tool_order)
        env.reset()
        return env.get_state()

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


        bot_location = state[0]
        if bot_location == TurtleBot.AT_WORK_ROOM:
            human_status = state[2]
            desired_tool = state[3]
            task_stage = self.correct_tool_order.index(desired_tool + 1) + 1
        else:
            human_status = 0
            task_stage = 0

        obs = [state[0], state[1], human_status, task_stage]

        return np.array(obs, dtype=int)

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

        prev_desired_tool = state[3]
        current_desired_tool = new_state[3]

        # human just receives a desired tool
        if prev_desired_tool != current_desired_tool:
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
        # the current desired tool is the final tool and human just receives it
        current_desired_tool = new_state[3]
        current_human_status = new_state[2]
        if current_desired_tool == self.correct_tool_order[-1] and current_human_status == Human.WORKING:
            done = True

        return done

    def step(self, action: int) -> DomainStepResult:
        """
        Combine all
        """
        env = EnvWareHouse(self.correct_tool_order)

        # reset to this state
        env.reset(self.state)

        sim_step = self.simulation_step(self.state, action)
        reward = self.reward(self.state, action, sim_step.state)
        terminal = self.terminal(self.state, action, sim_step.state)

        self.state = sim_step.state

        return DomainStepResult(sim_step.observation, reward, terminal)

class WareHousePrior(DomainPrior):
    """perfect prior for now
    """

    def __init__(self):
        """initiate the prior

        Args:
        """
        super().__init__()

    def sample(self, correct_prior=False) -> Domain:
        """
        """
        if correct_prior:
            correct_tool_order = [1, 2, 3, 4]
        else:
            perm = list(permutations([1, 2, 3, 4]))
            correct_tool_order = random.choice(perm)

        return WareHouse(list(correct_tool_order))

if __name__ == "__main__":
    env = WareHouse(correct_tool_order=[1, 2, 3, 4])
    env.reset()

    action_lst = ['Go_to_WR', 'Go_to_TR', 'Get_Tool_0', 'Get_Tool_1',
                'Get_Tool_2', 'Get_Tool_3', 'Deliver', 'No_Op']


    def action_to_str(action_idx):
        return f"Action: {action_lst[action_idx]}"

    action_list = [Go_to_TR, Get_tool_1, Go_to_WR, Deliver_tool, Go_to_TR, Get_tool_2, Go_to_WR, Deliver_tool]
    action_list += [Go_to_TR, Get_tool_3, Go_to_WR, Deliver_tool, Go_to_TR, Get_tool_4, Go_to_WR, Deliver_tool]

    for action in action_list:
        print(action_to_str(action))
        state = env.step(action)
    #     # print(state.reward)
        print(env.state)
        print(state.reward)
        print()
