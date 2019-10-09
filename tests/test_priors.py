""" tests the functionality of priors """

import random
import unittest

import numpy as np

from po_nrl.domains import Tiger, GridWorld, CollisionAvoidance
from po_nrl.domains.priors import TigerPrior, GridWorldPrior, CollisionAvoidancePrior
from po_nrl.environments import EncodeType


class TestTigerPrior(unittest.TestCase):
    """ tests that the observation probability is reasonable """

    def test_encoding(self) -> None:
        """ tests the sample method encoding is correct """

        one_hot_prior = TigerPrior(10, EncodeType.ONE_HOT)
        self.assertEqual(one_hot_prior.sample().observation_space.ndim, 2)

        default_prior = TigerPrior(10, EncodeType.DEFAULT)
        self.assertEqual(default_prior.sample().observation_space.ndim, 1)

    def test_observation_prob(self) -> None:
        """ tests the observation probability of samples """

        if random.choice([True, False]):
            encoding = EncodeType.DEFAULT
        else:
            encoding = EncodeType.ONE_HOT

        tiger = TigerPrior(10, encoding).sample()
        assert isinstance(tiger, Tiger)

        obs_probs = tiger._correct_obs_probs  # pylint: disable=protected-access
        self.assertTrue(0 <= obs_probs[0] <= 1)
        self.assertTrue(0 <= obs_probs[1] <= 1)

    def test_num_total_counts(self) -> None:
        # FIXME: NYI
        self.assertFalse(True)


class TestGridWorldPrior(unittest.TestCase):
    """ tests the prior over the gridworld problem """

    def test_encoding(self) -> None:
        """ tests encoding is done correctly """

        one_hot_sample = GridWorldPrior(size=3, encoding=EncodeType.ONE_HOT).sample()
        self.assertEqual(one_hot_sample.observation_space.ndim, 5)

        default_sample = GridWorldPrior(size=5, encoding=EncodeType.DEFAULT).sample()
        self.assertEqual(default_sample.observation_space.ndim, 3)

    def test_default_slow_cells(self) -> None:
        """ tests Gridworlds sampled from prior have no slow cells """

        sample_gridworld_1 = GridWorldPrior(size=4, encoding=EncodeType.DEFAULT).sample()
        sample_gridworld_2 = GridWorldPrior(size=4, encoding=EncodeType.DEFAULT).sample()

        assert isinstance(sample_gridworld_1, GridWorld)
        assert isinstance(sample_gridworld_2, GridWorld)

        self.assertTrue(sample_gridworld_1.slow_cells, "may **rarely** be empty")
        self.assertTrue(sample_gridworld_2.slow_cells, "may **rarely** be empty")
        self.assertTrue(sample_gridworld_1.slow_cells != sample_gridworld_2.slow_cells, "may **rarely** be true")


class TestCollisionAvoidancePrior(unittest.TestCase):
    """ tests the prior on the collection avoidance environment """

    def test_default(self) -> None:
        """ tests the default prior """

        sampled_domain = CollisionAvoidancePrior(3).sample()

        assert isinstance(sampled_domain, CollisionAvoidance)

        block_pol = sampled_domain._block_policy  # pylint: disable=protected-access

        np.testing.assert_array_less(block_pol, 1)
        np.testing.assert_array_less(0, block_pol)

        self.assertAlmostEqual(np.sum(block_pol), 1)
