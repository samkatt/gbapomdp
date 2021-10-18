"""tests :mod:`general_bayes_adaptive_pomdps.misc`"""

import random

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.misc import DiscreteSpace, set_random_seed


def test_num_elements():
    """tests whether the number of elements is as expected"""

    space = DiscreteSpace([3, 2, 3])

    assert space.n == space.num_elements
    assert space.n == 18


def test_num_dimensions():
    """ " tests whether it correctly returns the number of dimensions"""

    space = DiscreteSpace([3] * 8)

    assert space.ndim == 8


def test_sample():
    """tests sampling"""

    space = DiscreteSpace([5, 2])

    assert space.contains(space.sample())


def test_contain():
    """tests contains"""

    space = DiscreteSpace([2, 3])

    assert space.contains(np.array([0, 0]))
    assert not space.contains(np.array([-1, 0]))
    assert not space.contains(np.array([0, 3]))


def test_index_of():
    """tests getting index of things"""

    single_dim_space = DiscreteSpace([5])
    assert single_dim_space.index_of(np.array([0])) == 0
    assert single_dim_space.index_of(np.array([2])) == 2
    assert single_dim_space.index_of(np.array([4])) == 4

    multi_dim_space = DiscreteSpace([3, 2, 5])
    assert multi_dim_space.index_of(np.array([0, 0, 0])) == 0
    assert multi_dim_space.index_of(np.array([2, 0, 0])) == 2
    assert multi_dim_space.index_of(np.array([0, 1, 0])) == 3
    assert multi_dim_space.index_of(np.array([0, 0, 3])) == 18
    assert multi_dim_space.index_of(np.array([2, 0, 3])) == 20
    assert multi_dim_space.index_of(np.array([2, 1, 2])) == 17

    edge_case_space = DiscreteSpace([2, 1, 3])
    assert edge_case_space.index_of(np.array([0, 0, 2])) == 4


def test_space_from_index():
    """Tests index => element in :class:`DiscreteSpace`"""

    n = random.randint(2, 6)
    s = np.random.randint(2, 6, size=n)
    space = DiscreteSpace(s)

    with pytest.raises(AssertionError):
        space.from_index(-1)
    with pytest.raises(AssertionError):
        space.from_index(100000000)

    np.testing.assert_array_equal(space.from_index(0), np.zeros_like(n))

    for _ in range(10):
        elem = space.sample()
        idx = space.index_of(elem)

        np.testing.assert_array_equal(space.from_index(idx), elem)


def test_default_behaviour() -> None:
    """regular sampling"""

    random_sample = random.uniform(0, 1)
    assert round(abs(random_sample - random.uniform(0, 1)), 7) != 0

    random_np_sample = np.random.uniform(0, 1)
    assert round(abs(random_np_sample - np.random.uniform(0, 1)), 7) != 0


def test_setting_seed() -> None:
    """tests whether setting the seed will result in repetitive behaviour"""

    seed = random.randint(0, 1000)
    set_random_seed(seed)

    random_sample = random.uniform(0, 1)
    random_np_sample = np.random.uniform(0, 1)

    set_random_seed(seed)

    assert round(abs(random_sample - random.uniform(0, 1)), 7) == 0
    assert round(abs(random_np_sample - np.random.uniform(0, 1)), 7) == 0


if __name__ == "__main__":
    pytest.main([__file__])
