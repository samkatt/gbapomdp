"""Tests :mod:`general_bayes_adaptive_pomdps.domains`"""

import numpy as np
import pytest
import time

from general_bayes_adaptive_pomdps.domains.box_pushing.centralized_box_pushing import (
    CentralizedBoxPushing,
)

import one_to_one


sem_action_space = one_to_one.JointNamedSpace(
    a1=one_to_one.RangeSpace(4),  # 0: move forward, 1: turn left, 2: turn right, 3: stay
    a2=one_to_one.RangeSpace(4),
)
action_lst = list(sem_action_space.elems)

def setup_domain() -> CentralizedBoxPushing:
    """creates a env member"""
    domain = CentralizedBoxPushing()
    domain.reset()
    return domain


def test_reset():
    """tests that start state is 0"""

    env = setup_domain()

    # assert env.state in [21269]


if __name__ == "__main__":
    env = CentralizedBoxPushing(grid_dim=(4, 4), render=True)
    env.reset()

    done = False

    agent1_actions = [1, 0, 0]
    agent2_actions = [2, 0, 0]
    # agent1_actions = [1, 0, 0, 0, 0]

    i = 0

    while i < len(agent1_actions):
        action = action_lst[0]

        action.a1.value = agent1_actions[i]
        action.a2.value = agent2_actions[i]
        print(action.idx)
        # print(agent1_actions[i])
        result = env.step(action)
        done = result.terminal
        reward = result.reward
        print(reward)
        i += 1
        time.sleep(0.5)

        # if done:
        #     print("Reset")
        #     env.reset()
        #     done = False
