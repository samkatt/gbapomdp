"""Tests :mod:`general_bayes_adaptive_pomdps.domains`"""

import random

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.core import TerminalState
from general_bayes_adaptive_pomdps.domains.ma_box_pushing import (
    MultiAgentBoxPushing,
)
from general_bayes_adaptive_pomdps.misc import DiscreteSpace

import one_to_one


def setup_domain() -> MultiAgentBoxPushing:
    """creates a env member"""
    domain = MultiAgentBoxPushing()
    domain.reset()
    return domain


def test_reset():
    """tests that start state is 0"""

    env = setup_domain()

    assert env.state in [21269]


if __name__ == "__main__":
    pytest.main([__file__])
