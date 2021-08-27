"""Tests for :mod:`general_bayes_adaptive_pomdps.core`"""

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.core import ActionSpace


def test_action_space_one_hot():
    """Tests :meth:`ActionSpace.one_hot`"""
    action_space = ActionSpace(5)

    np.testing.assert_array_equal(action_space.one_hot(0), [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(action_space.one_hot(4), [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(action_space.one_hot(3), [0, 0, 0, 1, 0])


def test_action_sample_as_int():
    """Tests :meth:`ActionSpace.sample_as_int`"""
    size = 8
    action_space = ActionSpace(size)
    actions = {action_space.sample_as_int() for _ in range(1000)}
    assert actions == set(range(size))


if __name__ == "__main__":
    pytest.main([__file__])
