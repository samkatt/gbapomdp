""" tests the functionality of priors """

import random
import unittest

import numpy as np

from po_nrl.domains import Tiger, GridWorld, CollisionAvoidance, RoadRacer
from po_nrl.domains.priors import TigerPrior, GridWorldPrior
from po_nrl.domains.priors import CollisionAvoidancePrior, RoadRacerPrior
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
        """ tests the parameter # of total counts

        Args:

        RETURNS (`None`):

        """

        # randomly pick encoding
        if random.choice([True, False]):
            encoding = EncodeType.DEFAULT
        else:
            encoding = EncodeType.ONE_HOT

        tiger = TigerPrior(10000000, encoding).sample()
        assert isinstance(tiger, Tiger)

        obs_probs = tiger._correct_obs_probs  # pylint: disable=protected-access
        self.assertTrue(.59 <= obs_probs[0] <= .61)
        self.assertTrue(.59 <= obs_probs[1] <= .61)

        tiger = TigerPrior(1, encoding).sample()
        assert isinstance(tiger, Tiger)

        obs_probs = tiger._correct_obs_probs  # pylint: disable=protected-access
        self.assertFalse(.59 <= obs_probs[0] <= .61, f'Rarely false; obs = {obs_probs[0]}')
        self.assertFalse(.59 <= obs_probs[1] <= .61, f'Rarely false; obs = {obs_probs[0]}')


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

        sampled_domain = CollisionAvoidancePrior(3, 1).sample()

        assert isinstance(sampled_domain, CollisionAvoidance)

        block_pol = sampled_domain._block_policy  # pylint: disable=protected-access

        np.testing.assert_array_less(block_pol, 1)
        np.testing.assert_array_less(0, block_pol)

        self.assertAlmostEqual(np.sum(block_pol), 1)

    def test_certain_prior(self) -> None:
        """ very certain prior """
        sampled_domain = CollisionAvoidancePrior(3, 10000000).sample()

        assert isinstance(sampled_domain, CollisionAvoidance)

        block_pol = sampled_domain._block_policy  # pylint: disable=protected-access

        self.assertAlmostEqual(block_pol[0], .05, 3)
        self.assertAlmostEqual(block_pol[1], .9, 3)
        self.assertAlmostEqual(block_pol[2], .05, 3)

    def test_uncertain_prior(self) -> None:
        """ uncertain prior """
        sampled_domain = CollisionAvoidancePrior(3, 10).sample()

        assert isinstance(sampled_domain, CollisionAvoidance)

        block_pol = sampled_domain._block_policy  # pylint: disable=protected-access

        self.assertNotAlmostEqual(block_pol[0], .05, 6)
        self.assertNotAlmostEqual(block_pol[1], .9, 6)
        self.assertNotAlmostEqual(block_pol[2], .05, 6)


class TestRoadRacerPrior(unittest.TestCase):
    """ tests the road race prior """

    def test_lane_probs(self) -> None:
        """ some basic tests """

        # test .5 probabilities with certainty
        domain = RoadRacerPrior(3, 1000000000).sample()
        assert isinstance(domain, RoadRacer)

        np.testing.assert_almost_equal(domain.lane_probs, np.array([.5, .5, .5]), decimal=3)

        # test always within 0 and 1
        domain = RoadRacerPrior(5, .5).sample()
        assert isinstance(domain, RoadRacer)

        self.assertTrue(np.all(domain.lane_probs > 0), f'probs={domain.lane_probs}')
        self.assertTrue(np.all(domain.lane_probs < 1), f'probs={domain.lane_probs}')
