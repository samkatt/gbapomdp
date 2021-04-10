"""Tests for :mod:`general_bayes_adaptive_pomdps.domains`"""

import pytest

from general_bayes_adaptive_pomdps.domains import create_domain, create_prior
from general_bayes_adaptive_pomdps.domains.collision_avoidance import (
    CollisionAvoidance,
    CollisionAvoidancePrior,
)
from general_bayes_adaptive_pomdps.domains.gridworld import GridWorld, GridWorldPrior
from general_bayes_adaptive_pomdps.domains.road_racer import RoadRacer, RoadRacerPrior
from general_bayes_adaptive_pomdps.domains.tiger import Tiger, TigerPrior


@pytest.mark.parametrize(
    "string_input,domain_size,expected_domain",
    [
        ("tiger", 0, Tiger),
        ("gridworld", 5, GridWorld),
        ("collision_avoidance", 5, CollisionAvoidance),
        ("road_racer", 5, RoadRacer),
    ],
)
def test_create_domain(string_input, domain_size, expected_domain):
    assert isinstance(create_domain(string_input, domain_size), expected_domain)


def test_create_domain_raises():
    with pytest.raises(ValueError):
        create_domain("nonsense", 0)


@pytest.mark.parametrize(
    "string_input,domain_size,expected_prior",
    [
        ("tiger", 0, TigerPrior),
        ("gridworld", 5, GridWorldPrior),
        ("collision_avoidance", 5, CollisionAvoidancePrior),
        ("road_racer", 5, RoadRacerPrior),
    ],
)
def test_create_prior(string_input, domain_size, expected_prior):
    assert isinstance(
        create_prior(string_input, domain_size, 10, 0, False), expected_prior
    )


def test_create_prior_raises():
    with pytest.raises(ValueError):
        create_prior("nonsense", 0, 10, 0, False)


if __name__ == "__main__":
    pytest.main([__file__])
