"""Tests :mod:`general_bayes_adaptive_pomdps.models.baddr"""

import random
from functools import partial

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.domains.tiger import Tiger
from general_bayes_adaptive_pomdps.models.baddr import (
    create_model_updates,
    get_model_freeze_setting,
    sample_transitions_uniform,
    train_from_samples,
)
from general_bayes_adaptive_pomdps.models.neural_networks.neural_pomdps import (
    DynamicsModel,
    sgd_builder,
)


class TestTrainFromSamples:
    """Basic test for learning ensembles from samples"""

    def test_improve_performance(self):
        """Tests whether learning improves model"""
        sim = Tiger(one_hot_encode_observation=False)
        sampler = partial(
            sample_transitions_uniform,
            state_space=sim.state_space,
            action_space=sim.action_space,
            domain_simulation_step=sim.simulation_step,
        )

        batch_size = 16

        t_net = DynamicsModel.TNet(
            sim.state_space,
            sim.action_space,
            sgd_builder,
            learning_rate=0.1,
            network_size=16,
            dropout_rate=0.0,
        )

        o_net = DynamicsModel.ONet(
            sim.state_space,
            sim.action_space,
            sim.observation_space,
            sgd_builder,
            learning_rate=0.1,
            network_size=16,
            dropout_rate=0.0,
        )

        model = DynamicsModel(
            sim.state_space,
            sim.action_space,
            batch_size=batch_size,
            t_model=t_net,
            o_model=o_net,
        )

        s = np.array([random.randint(0, 1)])

        initial_transition_model = model.transition_model(s, Tiger.LISTEN)[0]

        train_from_samples(
            model,
            sampler,
            num_epochs=256,
            batch_size=batch_size,
        )

        final_transition_model = model.transition_model(s, Tiger.LISTEN)[0]

        # the model should have learned that the transition probability of staying in the same state is high
        assert initial_transition_model[s[0]] <= final_transition_model[s[0]]


class TestSampleFromSimulator:
    """Tests `sample_transitions_uniform_from_simulator`"""

    def test_simple_run_and_space(self):
        """run once on tiger and check if result is viable"""
        sim = Tiger(one_hot_encode_observation=False)

        (s, a, news, o,) = sample_transitions_uniform(
            sim.state_space, sim.action_space, sim.simulation_step
        )

        assert sim.state_space.contains(s)
        assert sim.action_space.contains(a)
        assert sim.state_space.contains(news)
        assert sim.observation_space.contains(o)


@pytest.mark.parametrize(
    "string_input,expected_setting",
    [
        ("", DynamicsModel.FreezeModelSetting.FREEZE_NONE),
        ("T", DynamicsModel.FreezeModelSetting.FREEZE_T),
        ("O", DynamicsModel.FreezeModelSetting.FREEZE_O),
    ],
)
def test_get_model_freeze_setting(string_input, expected_setting):
    assert get_model_freeze_setting(string_input) == expected_setting


def test_get_model_freeze_setting_raises():
    with pytest.raises(ValueError):
        get_model_freeze_setting("nonsense")


@pytest.mark.parametrize(
    "backprop,replay,perturb,len_output",
    [
        (False, False, 0, 0),
        (True, False, 0, 1),
        (True, True, 0, 2),
        (False, True, 0, 1),
        (False, True, 0.3, 2),
        (True, True, 0.3, 3),
    ],
)
def test_create_model_updates(backprop, replay, perturb, len_output):
    assert (
        len(
            create_model_updates(
                DynamicsModel.FreezeModelSetting.FREEZE_NONE, backprop, replay, perturb
            )
        )
        == len_output
    )


if __name__ == "__main__":
    pytest.main([__file__])
