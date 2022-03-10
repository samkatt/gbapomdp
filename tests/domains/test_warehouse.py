"""Tests :mod:`general_bayes_adaptive_pomdps.domains`"""

import random

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.core import TerminalState
from general_bayes_adaptive_pomdps.domains.warehouse import (
    WareHouse,
    create_tabular_prior_counts,
)
from general_bayes_adaptive_pomdps.misc import DiscreteSpace

import one_to_one

# Position status
AT_TOOL_ROOM = 0
AT_WORK_ROOM = 1

# Tool status
TOOL_AVAIL   = 0
TOOL_PICKED  = 1
TOOL_DROPPED = 2

# Actions
PICK_0 = 0
PICK_1 = 1
PICK_2 = 2
DROP   = 3
NUM_TOOLS = 3

# Human status
UNKNOWN_STATUS = 4

# Rewards
STEP_REWARD          = -1
TASK_FINISHED_REWARD = 100
WRONG_TOOL_REWARD    = -10
STEP_CORRECT_TOOL_REWARD    = 10

_max_human_status = 4
_max_tool_status = 3
_max_location = 2

state_space = one_to_one.JointNamedSpace(
    human_status=one_to_one.RangeSpace(_max_human_status),

    tool_status_0=one_to_one.RangeSpace(_max_tool_status),
    tool_status_1=one_to_one.RangeSpace(_max_tool_status),
    tool_status_2=one_to_one.RangeSpace(_max_tool_status),

    location=one_to_one.RangeSpace(_max_location)
    )
p_state_space_lst = list(state_space.elems)


def setup_warehouse(one_hot: bool) -> WareHouse:
    """creates a env member"""
    domain = WareHouse(one_hot_encode_observation=one_hot)
    domain.reset()
    return domain


def return_state(index: int) -> one_to_one.JointNamedSpace:
    return p_state_space_lst[index]


def check_state(idx, human_status, tool_status_0, tool_status_1, tool_status_2, location):
    ok = True

    state = return_state(idx[0])

    if state.human_status.value != human_status or state.tool_status_0.value != tool_status_0 \
        or state.tool_status_1.value != tool_status_1 or state.tool_status_2.value != tool_status_2 or \
            state.location.value != location:
        ok = False

    return ok


def test_reset():
    """tests that start state is 0"""

    env = setup_warehouse(False)

    assert env.state in [0]


def test_drop():
    """tests if the episode resets after a drop"""

    env = setup_warehouse(False)

    step = env.step(DROP)

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 0, TOOL_AVAIL, TOOL_AVAIL, TOOL_AVAIL, AT_TOOL_ROOM))


def test_drop_wrong_tool():
    """Test if the episode resets after dropping a wrong tool"""
    env = setup_warehouse(False)

    step = env.step(PICK_1)

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 0, TOOL_AVAIL, TOOL_PICKED, TOOL_AVAIL, AT_TOOL_ROOM))

    step = env.step(DROP)

    assert step.terminal
    assert step.reward == WRONG_TOOL_REWARD
    assert (check_state(step.observation, 0, TOOL_AVAIL, TOOL_DROPPED, TOOL_AVAIL, AT_WORK_ROOM))

    obs = env.reset()

    assert (check_state(obs, 0, TOOL_AVAIL, TOOL_AVAIL, TOOL_AVAIL, AT_TOOL_ROOM))

    step = env.step(PICK_0)

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 0, TOOL_PICKED, TOOL_AVAIL, TOOL_AVAIL, AT_TOOL_ROOM))

    step = env.step(DROP)
    state = return_state(step.observation[0])

    assert not step.terminal
    assert step.reward == STEP_CORRECT_TOOL_REWARD
    assert (check_state(step.observation, 1, TOOL_DROPPED, TOOL_AVAIL, TOOL_AVAIL, AT_WORK_ROOM))


def test_double_picks():

    env = setup_warehouse(False)

    step = env.step(PICK_0)
    state = return_state(step.observation[0])

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 0, TOOL_PICKED, TOOL_AVAIL, TOOL_AVAIL, AT_TOOL_ROOM))

    step = env.step(PICK_1)
    state = return_state(step.observation[0])

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 0, TOOL_PICKED, TOOL_AVAIL, TOOL_AVAIL, AT_TOOL_ROOM))


def test_step():
    """tests some basic steps"""

    env = setup_warehouse(False)

    # Test terminal
    step = env.reset()

    step = env.step(PICK_1)
    state = return_state(step.observation[0])

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 0, TOOL_AVAIL, TOOL_PICKED, TOOL_AVAIL, AT_TOOL_ROOM))

    step = env.step(DROP)
    state = return_state(step.observation[0])

    assert step.reward == WRONG_TOOL_REWARD
    assert (check_state(step.observation, 0, TOOL_AVAIL, TOOL_DROPPED, TOOL_AVAIL, AT_WORK_ROOM))

    step = env.reset()
    step = env.step(PICK_2)

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 0, TOOL_AVAIL, TOOL_AVAIL, TOOL_PICKED, AT_TOOL_ROOM))

    step = env.step(DROP)
    assert step.terminal
    assert step.reward == WRONG_TOOL_REWARD
    assert (check_state(step.observation, 0, TOOL_AVAIL, TOOL_AVAIL, TOOL_DROPPED, AT_WORK_ROOM))

    # Test optimal policy
    step = env.reset()
    step = env.step(PICK_0)

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 0, TOOL_PICKED, TOOL_AVAIL, TOOL_AVAIL, AT_TOOL_ROOM))

    step = env.step(DROP)

    assert not step.terminal
    assert step.reward == STEP_CORRECT_TOOL_REWARD
    assert (check_state(step.observation, 1, TOOL_DROPPED, TOOL_AVAIL, TOOL_AVAIL, AT_WORK_ROOM))

    step = env.step(PICK_1)

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 1, TOOL_DROPPED, TOOL_PICKED, TOOL_AVAIL, AT_TOOL_ROOM))

    step = env.step(DROP)

    assert not step.terminal
    assert step.reward == STEP_CORRECT_TOOL_REWARD
    assert (check_state(step.observation, 2, TOOL_DROPPED, TOOL_DROPPED, TOOL_AVAIL, AT_WORK_ROOM))

    step = env.step(PICK_2)

    assert not step.terminal
    assert step.reward == STEP_REWARD
    assert (check_state(step.observation, 2, TOOL_DROPPED, TOOL_DROPPED, TOOL_PICKED, AT_TOOL_ROOM))

    step = env.step(DROP)

    assert step.terminal
    assert step.reward == TASK_FINISHED_REWARD
    assert (check_state(step.observation, 3, TOOL_DROPPED, TOOL_DROPPED, TOOL_DROPPED, AT_WORK_ROOM))


if __name__ == "__main__":
    pytest.main([__file__])
