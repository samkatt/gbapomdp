"""Tests :mod:`general_bayes_adaptive_pomdps.baddr.model"""

import random
from functools import partial

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.baddr.model import (
    create_transition_sampler,
    sample_transitions_uniform_from_simulator,
    train_from_samples,
)
from general_bayes_adaptive_pomdps.baddr.neural_networks.neural_pomdps import (
    DynamicsModel,
    sgd_builder,
)
from general_bayes_adaptive_pomdps.domains import Tiger


class TestTrainFromSamples:
    """Basic test for learning ensembles from samples"""

    def test_improve_performance(self):
        """Tests whether learning improves model"""
        sim = Tiger(one_hot_encode_observation=False)
        sampler = sample_transitions_uniform_from_simulator

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
            partial(sampler, sim=sim),
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

        (
            s,
            a,
            news,
            o,
        ) = sample_transitions_uniform_from_simulator(sim)

        assert sim.state_space.contains(s)
        assert sim.action_space.contains(a)
        assert sim.state_space.contains(news)
        assert sim.observation_space.contains(o)


class TestCreateTransitionSampler:
    """Tests factory for transition samplers

    This test assumes (knows) that the factory function returns a partial.  The
    name of that partial is being checked. Although silly, it catches the
    otherwise hard-to-notice evil bug

    """

    def test_default(self):
        """Test none-Gridverse """
        assert (
            create_transition_sampler(None).func.__name__  # type: ignore
            == "sample_transitions_uniform_from_simulator"
        )


if __name__ == "__main__":
    pytest.main([__file__])
