"""The tiger problem implemented as domain"""

from typing import Tuple
from general_bayes_adaptive_pomdps.domains.warehouse.env_warehouse import TurtleBot

import numpy as np

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
    TerminalState,
    InvalidState
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace

from general_bayes_adaptive_pomdps.domains.warehouse_v0.env_warehouse import EnvWareHouse, Human
import one_to_one


class WareHouse(Domain):

    def __init__(self, render=False):
        """Construct the tiger domain
        Args:
            grid_dim

        """
        super().__init__()

        self.n_turtlebots = 1

        # state: (location, current-tool) per turtlebot + (human-status, desired-tool, human-step-cnt)
        # location: 0 (tool-room) or 1 (work-room) (2)
        # current-tool: 0, 1, 2, 3, 4 (5)
        # human-status: working or waiting (2)
        # desired-tool: 0, 1, 2, 3, 4 (5)
        # human step count: 0, 1, 2 (3)
        self._state_space = DiscreteSpace([2, 5]*self.n_turtlebots + [2, 5] + [3])

        # obs: (location, current-tool) per turtlebot + (human-status)
        # location: tool-room or work-room (2)
        # current-tool: 0, 1, 2, 3, 4 (5)
        # human-status: working, waiting, NULL (3) (only available when one turtlebot is at the work room)
        # desired-tool: 1, 2, 3, 4, 5 (5) (only available when one turtlebot is at the work room)
        self._obs_space = DiscreteSpace([2, 5]*self.n_turtlebots + [3, 5])

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

        RETURNS (`np.ndarray`): [x, y, ori] per agent, [x, y] per box

        """
        env = EnvWareHouse()
        env.reset()
        return env.get_state()

    def generate_observation(self, state: np.ndarray = None) -> np.ndarray:
        """generates an observation of the state (noisy obstacle sensor)

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
        # desired-tool: 1, 2, 3, 4, 5 (5)


        bot_location = state[0]
        if bot_location == TurtleBot.AT_WORK_ROOM:
            human_status = state[2]
            desired_tool = state[3]
        else:
            human_status = 2  # NULL
            desired_tool = 0

        obs = [state[0], state[1], human_status, desired_tool]

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

        env = EnvWareHouse()

        # reset to this state
        env.reset(state)

        if env.is_terminal_state(state):
            raise TerminalState(f"state {state} is terminal")

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
        if prev_desired_tool + 1 == current_desired_tool:
            reward += 10

        # human finished using the final tool
        # if current_desired_tool == 4 and current_human_time == 2:
            # reward += 100

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
        # the current desired tool is the final tool
        # the agent finishes using that tool
        current_desired_tool = new_state[3]
        current_human_time = new_state[4]
        if current_desired_tool == 4 and current_human_time == 2:
            done = True

        return done

    def step(self, action: int) -> DomainStepResult:
        """
        Combine all
        """
        env = EnvWareHouse()

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

    def sample(self) -> Domain:
        """
        """
        return WareHouse()
