"""Tests the Tabular BA-POMDP implementation"""
from collections import Counter

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.models.tabular_bapomdp import (
    expected_model,
    expected_probability,
    sample_large_dirichlet,
    sample_small_dirichlet,
)


@pytest.mark.parametrize(
    "counts,expected_val",
    [
        ([1], 0),
        ([10], 0),
        ([0, 1], 1),
        ([1, 0], 0),
        ([1, 1, 1, 1, 1, 10000, 1, 1, 1], 5),
    ],
)
def test_sample_dirichlet_corner(counts, expected_val):
    """Tests output on predictable corner cases of :func:`sample_small_dirichlet`"""
    assert sample_small_dirichlet(np.array(counts)) == expected_val
    assert sample_large_dirichlet(np.array(counts)) == expected_val


@pytest.mark.parametrize(
    "counts",
    [
        ([1, 1, 1]),
        ([100, 100, 100]),
        ([123, 123, 123, 123, 123, 123, 123]),
        ([100, 80, 90, 120]),
    ],
)
def test_sample_dirichlet_all(counts):
    """Tests output on predictable uniform case of :func:`sample_small_dirichlet`"""
    samples = {sample_small_dirichlet(np.array(counts)) for _ in range(100)}
    assert {*range(len(counts))} == samples

    samples = {sample_large_dirichlet(np.array(counts)) for _ in range(100)}
    assert {*range(len(counts))} == samples


@pytest.mark.parametrize(
    "counts,max_elem,min_elem",
    [
        ([1, 10, 5], 1, 0),
        ([100, 50, 25], 0, 2),
        ([10, 5, 5, 0.5, 5, 20, 5], 5, 3),
    ],
)
def test_sample_dirichlet_max_min(counts, max_elem, min_elem):
    """Tests statistics of output on :func:`sample_small_dirichlet`"""

    for f in [sample_small_dirichlet, sample_large_dirichlet]:
        samples = Counter(f(np.array(counts)) for _ in range(500))

        ordered = samples.most_common()

        assert ordered[0][0] == max_elem
        assert ordered[-1][0] == min_elem


@pytest.mark.parametrize(
    "counts,probs",
    [([1, 1], [0.5, 0.5]), ([2, 2], [0.5, 0.5]), ([1.5, 13.5], [0.1, 0.9])],
)
def test_expected_model(counts, probs):
    """Tests :func:`expected_model`"""
    np.testing.assert_allclose(expected_model(counts), np.array(probs))


@pytest.mark.parametrize(
    "counts,i,prob",
    [
        ([1, 1], 0, 0.5),
        ([1, 1], 1, 0.5),
        ([1.5, 1.5, 12], 1, 0.1),
        ([1.5, 1.5, 12], 2, 0.8),
    ],
)
def test_expected_probability(counts, i, prob):
    """Tests :func:`expected_probability`"""
    assert expected_probability(counts, i) == prob
