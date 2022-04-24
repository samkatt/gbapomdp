"""Tests :mod:`general_bayes_adaptive_pomdps.domains`"""

import numpy as np
import pytest
import time

from general_bayes_adaptive_pomdps.domains.box_pushing.centralized_box_pushing import (
    CentralizedBoxPushing,
)


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
    env = CentralizedBoxPushing()
    env.reset()

    done = False

    while not done:
        action = np.random.randint(16)
        result = env.step(action)
        done = result.terminal
        time.sleep(0.5)

        if done:
            print("Reset")
            env.reset()
            done = False
