"""runs tests on the functionality in `learned_environmets.py`"""
import random
import unittest
from functools import partial

import numpy as np
from po_nrl.agents.neural_networks.neural_pomdps import (DynamicsModel,
                                                         sgd_builder)
from po_nrl.domains import GridverseDomain, Tiger
from po_nrl.domains.learned_environments import (
    create_transition_sampler, sample_from_gridverse,
    sample_transitions_uniform_from_simulator, train_from_samples)
from po_nrl.environments import EncodeType


class TestSampleFromSimulator(unittest.TestCase):
    """Tests `sample_transitions_uniform_from_simulator`"""

    def test_simple_run_and_space(self):
        """run once on tiger and check if result is viable"""
        sim = Tiger(EncodeType.DEFAULT)

        s, a, news, o = sample_transitions_uniform_from_simulator(sim)

        self.assertTrue(sim.state_space.contains(s))
        self.assertTrue(sim.action_space.contains(a))
        self.assertTrue(sim.state_space.contains(news))
        self.assertTrue(sim.observation_space.contains(o))


class TestTrainFromSamples(unittest.TestCase):
    """Basic test for learning ensembles from samples"""

    def test_improve_performance(self):
        """Tests whether learning in tiger environment improves model"""
        sim = Tiger(EncodeType.DEFAULT)
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
        self.assertLessEqual(
            initial_transition_model[s[0]], final_transition_model[s[0]]
        )


class TestSampleFromGridverse(unittest.TestCase):
    """Tests generating transitions from gridverse"""

    def test_that_it_runs(self):
        """basic call to ensure it does not crash"""
        d = GridverseDomain()

        try:
            _, _, _, _ = sample_from_gridverse(d)
        except Exception as e:  # pylint: disable=broad-except
            self.fail(f"This code should run, causes {e}")


class TestCreateTransitionSampler(unittest.TestCase):
    """Tests factory for transition samplers

    This test assumes (knows) that the factory function returns a partial.  The
    name of that partial is being checked. Although silly, it catches the
    otherwise hard-to-notice evil bug

    """

    # pylint: disable=no-member

    def test_default(self):
        """Test none-Gridverse """
        self.assertEqual(
            create_transition_sampler(None).func.__name__,  # type: ignore
            'sample_transitions_uniform_from_simulator',
        )

    def test_gridverse(self):
        """Test Gridverse """
        self.assertEqual(
            create_transition_sampler(GridverseDomain()).func.__name__,  # type: ignore
            'sample_from_gridverse',
        )


if __name__ == '__main__':
    unittest.main()
