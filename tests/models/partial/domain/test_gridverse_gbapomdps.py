""" tests the functionality of priors """

import pytest
import torch
from gym_gridverse.envs.factory import env_from_descr
from gym_gridverse.envs.gridworld import GridWorld as GVerseGridworld
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)

from general_bayes_adaptive_pomdps.models.partial.domain.gridverse_gbapomdps import (
    GridversePositionAugmentedState,
    agent_position,
    create_gridverse_prior,
    gverse_obs2array,
)


@pytest.mark.parametrize("pos", [((5, 2)), ((0, 0)), ((100, 50))])
def test_agent_position(pos):
    d = env_from_descr("KeyDoor-16x16-v0")
    s = d.functional_reset()
    s.agent.position = pos

    assert (agent_position(s) == pos).all()


def test_gverse_obs2array():
    d = env_from_descr("Dynamic-Obstacles-6x6-v0")
    s = d.functional_reset()

    obs = gverse_obs2array(d, DefaultObservationRepresentation(d.observation_space), s)
    assert obs.shape == (297,)


def model_equals(model_a, model_b) -> bool:
    """returns whether provided pytorch models are equal"""
    for tensor_a, tensor_b in zip(model_a.parameters(), model_b.parameters()):
        if not torch.equal(tensor_a.data, tensor_b.data):
            return False

    return True


def test_create_gridverse_prior():

    d = env_from_descr("Empty-5x5-v0")
    assert isinstance(d, GVerseGridworld)

    opt = "SGD"
    alpha = 0.1
    s = 32
    drop = 0.5
    n = 128
    batch = 8

    p = create_gridverse_prior(d, opt, alpha, s, drop, n, batch)

    s = p()
    a = 0

    assert isinstance(s, GridversePositionAugmentedState)

    next_s, o = s.domain_step(a)
    assert isinstance(next_s, GridversePositionAugmentedState)
    assert not s.domain_state == next_s.domain_state

    assert model_equals(s.learned_model.net, next_s.learned_model.net)

    next_s = s.update_model_distribution(s, a, next_s, o, optimize=True)
    assert isinstance(next_s, GridversePositionAugmentedState)
    assert model_equals(s.learned_model.net, next_s.learned_model.net)

    next_s = s.update_model_distribution(s, a, next_s, o)
    assert isinstance(next_s, GridversePositionAugmentedState)
    assert not model_equals(s.learned_model.net, next_s.learned_model.net)
