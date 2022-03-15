"""The warehouse problem implemented as domain"""

from logging import Logger
from random import random
from typing import List, Optional
import itertools as itt
import numpy as np
from copy import copy
import one_to_one

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain
from general_bayes_adaptive_pomdps.misc import DiscreteSpace, LogLevel
from general_bayes_adaptive_pomdps.models.tabular_bapomdp import DirCounts

# Position status
AT_TOOL_ROOM = 0
AT_WORK_ROOM = 1

# Tool status
TOOL_AVAIL = 0
TOOL_PICKED = 1
TOOL_DROPPED = 2

# Actions
PICK_0 = 0
PICK_1 = 1
PICK_2 = 2
DROP = 3

NUM_TOOLS = 3

# Human status
UNKNOWN_STATUS = 4

# Rewards
STEP_REWARD = -1
TASK_FINISHED_REWARD = 100
WRONG_TOOL_REWARD = -10
STEP_CORRECT_TOOL_REWARD = 10


class WareHouseV2(Domain):
    """Now the agent can only observe human status when at the work room,
    when at the tool room, it observer UNKNOWN_STATUS.
    It also can only observe which tool it is carrying instead of the status 
    of all tools
    """

    def __init__(
        self,
        one_hot_encode_observation: bool,
    ):
        """Construct the tiger domain

        Args:
             one_hot_encode_observation: (`bool`):
             correct_obs_probs: (`Optional[List[float]]`):

        """
        super().__init__()

        self._logger = Logger(self.__class__.__name__)

        self._use_one_hot_obs = False

        self._action_space = ActionSpace(NUM_TOOLS + 1)  # Drop, Pick_i for NUM_TOOLS tools

        self._correct_human_stat_probs = 0.8

        self._max_human_status = 5
        self._max_tool_status = 3
        self._max_location = 2

        # Human Status: [0, 1, 2, 3] x Tool-i Status: [0, 1, 2] x Agent's location [0, 1] 
        num_states = (self._max_human_status - 1) * self._max_tool_status**NUM_TOOLS * self._max_location

        # Human Status: [0, 1, 2, 3, UNKNOWN=4] x [Tool being carried (0, 1, 2, 3-Not carrying)] x Agent's location [0, 1] 
        num_obss = self._max_human_status * (NUM_TOOLS + 1) * self._max_location

        self._obs_space = DiscreteSpace([num_obss])
        self._state_space = DiscreteSpace([num_states])

        # For generating the true transition and observation tables
        self.sem_state_space = one_to_one.JointNamedSpace(
            human_status=one_to_one.RangeSpace(self._max_human_status - 1),

            tool_status_0=one_to_one.RangeSpace(self._max_tool_status),
            tool_status_1=one_to_one.RangeSpace(self._max_tool_status),
            tool_status_2=one_to_one.RangeSpace(self._max_tool_status),

            location=one_to_one.RangeSpace(self._max_location)
        )
        self._state_lst = list(self.sem_state_space.elems)

        self.sem_obs_space = one_to_one.JointNamedSpace(
            human_status=one_to_one.RangeSpace(self._max_human_status),
            tool_being_carried=one_to_one.RangeSpace(NUM_TOOLS + 1),
            location=one_to_one.RangeSpace(self._max_location)
        )
        self._obs_lst = list(self.sem_obs_space.elems)

        self.sem_action_space = one_to_one.RangeSpace(NUM_TOOLS + 1)

        self._state = self.sample_start_state()

    @property
    def state(self):
        """returns current state"""
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """sets state

        Args:
             state: (`np.ndarray`):

        """
        self._state = state

    @property
    def state_space(self) -> DiscreteSpace:
        """a `general_bayes_adaptive_pomdps.misc.DiscreteSpace` space"""
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        """a `general_bayes_adaptive_pomdps.core.ActionSpace` space"""
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """a `general_bayes_adaptive_pomdps.misc.DiscreteSpace` """
        return self._obs_space

    @staticmethod
    def sample_start_state() -> np.ndarray:
        """samples a random state (tiger left or right)

        RETURNS (`np.narray`): an initial state

        """
        # Human starts assembling, all tools are available, the agent is at the tool room
        return np.array([0], dtype=int)

    def reset(self) -> np.ndarray:
        """Resets internal state and return first observation

        """
        self._state = self.sample_start_state()

        return np.array([0], dtype=int)

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """Simulates stepping from state using action. Returns interaction

        Will terminate episode when a wrong tool is delivered or the task finishes,
        otherwise return an observation.

        Args:
             state: 
             action: 

        RETURNS (`general_bayes_adaptive_pomdps.core.SimulationResult`): the transition

        """

        _state = self._state_lst[state[0]]

        _new_state = copy(_state)

        num_tools_picked = numToolsPicked(_state)
        assert(num_tools_picked <= 1), "Error, agent carries more than 1 tool"

        if action in [DROP]:

            if num_tools_picked == 1:

                # This action brings the agent to the work room
                if _state.location.value == AT_TOOL_ROOM:
                    _new_state.location.value = AT_WORK_ROOM

                # Update status of the tool that is carried
                picked_tool_idx = toolPicked(_state)

                # Change the status of the tool that will be dropped
                if (picked_tool_idx == 0):
                    assert(_new_state.tool_status_0.value == TOOL_PICKED)
                    _new_state.tool_status_0.value = TOOL_DROPPED

                elif (picked_tool_idx == 1):
                    assert(_new_state.tool_status_1.value == TOOL_PICKED)
                    _new_state.tool_status_1.value = TOOL_DROPPED

                elif (picked_tool_idx == 2):
                    assert(_new_state.tool_status_2.value == TOOL_PICKED)
                    _new_state.tool_status_2.value = TOOL_DROPPED

                else:
                    print("Invalid picked tool index")

                # The agent drops the desired tool
                if _state.human_status.value == picked_tool_idx:
                    _new_state.human_status.value = _state.human_status.value + 1

        elif action in [PICK_0, PICK_1, PICK_2]:

            if num_tools_picked == 0:
                # If pick action is performed, the agent will be at the tool room
                if _state.location.value == AT_WORK_ROOM:
                    _new_state.location.value = AT_TOOL_ROOM

                if action == PICK_0:
                    # only change the status of a tool if it has not been picked before and the agent 
                    # is carrying no tool
                    if _state.tool_status_0.value == TOOL_AVAIL:
                        _new_state.tool_status_0.value = TOOL_PICKED

                if action == PICK_1:
                    if _state.tool_status_1.value == TOOL_AVAIL:
                        _new_state.tool_status_1.value = TOOL_PICKED

                if action == PICK_2:
                    if _state.tool_status_2.value == TOOL_AVAIL:
                        _new_state.tool_status_2.value = TOOL_PICKED

        else:
            print("Invalid action", action)

        assert(isStateValid(_new_state))
        obs = self._prepare_obs(_new_state)

        return SimulationResult(np.array([_new_state.idx]), np.array([obs.idx]))

    def _prepare_obs(self, state):
        obs = self._obs_lst[0]

        obs.location.value = state.location.value

        # The agent cannot observe human status at tool room
        if obs.location.value == AT_TOOL_ROOM:
            obs.human_status.value = UNKNOWN_STATUS
        else:
            # At work room, can observe human status with some prob.
            if np.random.random() < self._correct_human_stat_probs:
                obs.human_status.value = state.human_status.value
            else:
                obs.human_status.value = UNKNOWN_STATUS

        num_tools_picked = numToolsPicked(state)

        if num_tools_picked == 1:
            obs.tool_being_carried.value = toolPicked(state)
        else:
            assert(num_tools_picked == 0)
            # carrying no tool
            obs.tool_being_carried.value = NUM_TOOLS

        obs.location.value = state.location.value

        return obs

    def step(self, action: int) -> DomainStepResult:
        """Performs a step in the tiger problem given action

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             action: (`int`): 0 is open left, 1 is open right or 2 is listen

        RETURNS (`general_bayes_adaptive_pomdps.core.EnvironmentInteraction`): the transition

        """

        sim_result = self.simulation_step(self.state, action)
        reward = self.reward(self.state, action, sim_result.state)
        terminal = self.terminal(self.state, action, sim_result.state)

        self.state = sim_result.state

        return DomainStepResult(sim_result.observation, reward, terminal)

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """A constant if listening, penalty if opening to door, and reward otherwise

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`):

        """

        assert self.state_space.contains(state), f"{state} not in space"
        assert self.state_space.contains(new_state), f"{new_state} not in space"
        assert self.action_space.contains(action), f"{action} not in space"

        curr_state = self._state_lst[new_state[0]]
        prev_state = self._state_lst[state[0]]

        assert(isStateValid(curr_state))
        assert(isStateValid(prev_state))

        num_tools_picked = numToolsPicked(prev_state)
        assert(num_tools_picked <= 1), "Error, agent carries more than 1 tool"

        if action in [DROP]:

            if num_tools_picked == 1:
                picked_tool_idx = toolPicked(prev_state)

                if prev_state.human_status.value == picked_tool_idx:
                    if curr_state.human_status.value == self._max_human_status - 2:
                        return TASK_FINISHED_REWARD
                    else:
                        return STEP_CORRECT_TOOL_REWARD
                else:
                    return WRONG_TOOL_REWARD
            else:
                return STEP_REWARD

        else:
            return STEP_REWARD

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if opening a door

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`):

        """

        assert self.state_space.contains(state), f"{state} not in space"
        assert self.state_space.contains(new_state), f"{new_state} not in space"
        assert self.action_space.contains(action), f"{action} not in space"

        curr_state = self._state_lst[new_state[0]]
        prev_state = self._state_lst[state[0]]

        num_tools_picked = numToolsPicked(prev_state)
        assert(num_tools_picked <= 1), "Error, agent carries more than 1 tool"

        if action in [DROP]:
            if num_tools_picked == 1:
                picked_tool_idx = toolPicked(prev_state)

                if prev_state.human_status.value == picked_tool_idx:
                    if curr_state.human_status.value == self._max_human_status - 2:
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                return False

        else:
            return False


def isStateValid(s) -> bool:
    """A valid state cannot carries more 2 tools at a time"""

    return (numToolsPicked(s) <= 1)


def isObsValid(o) -> bool:
    cond2 = True
    if o.location.value == AT_TOOL_ROOM:
        cond2 = (o.human_status.value == UNKNOWN_STATUS)

    return cond2


def numToolsPicked(s) -> int:
    """Return # tools that are currently picked"""

    tool_picked = [s.tool_status_0.value == TOOL_PICKED,
                   s.tool_status_1.value == TOOL_PICKED,
                   s.tool_status_2.value == TOOL_PICKED]

    num_tools_picked = np.sum(tool_picked)

    return num_tools_picked


def toolPicked(s) -> int:
    """Return which tool is currently picked"""

    assert(isStateValid(s))

    tool_picked = np.array([s.tool_status_0.value == TOOL_PICKED,
                            s.tool_status_1.value == TOOL_PICKED,
                            s.tool_status_2.value == TOOL_PICKED])

    return np.argwhere(tool_picked)[0][0]


def create_tabular_prior_counts(
    correctness: float = 1, certainty: float = 10
) -> DirCounts:

    env = WareHouseV2(False)
    env.reset()

    state_space = env.sem_state_space
    obs_space = env.sem_obs_space
    action_space = env.sem_action_space

    T_mat = np.zeros((state_space.nelems,  action_space.nelems, state_space.nelems))
    O_mat = np.ones((action_space.nelems, state_space.nelems,  obs_space.nelems))

    for s in state_space.elems:
        for a in action_space.elems:
            if isStateValid(s):
                simulation_result = env.simulation_step([s.idx], a.idx)
                next_s_idx = simulation_result.state
                next_o_idx = simulation_result.observation

                next_s = env._state_lst[next_s_idx[0]]
                next_o = env._obs_lst[next_o_idx[0]]

                if isStateValid(next_s):
                    T_mat[s.idx, a.idx, next_s_idx[0]] = 1.0
                    # O_mat[a.idx, next_s_idx[0], next_o_idx[0]] = 1.0

    return DirCounts((1000*T_mat).astype(np.int), (1000*O_mat).astype(np.int))
