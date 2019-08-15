""" tests the functionality of priors """

import unittest

import random

from domains import Tiger, GridWorld
from domains.priors import TigerPrior, GridWorldPrior
from environments import EncodeType


class TestTigerPrior(unittest.TestCase):
    """ tests that the observation probability is reasonable """

    def test_encoding(self) -> None:
        """ tests the sample method encoding is correct """

        one_hot_prior = TigerPrior(EncodeType.ONE_HOT)
        self.assertEqual(one_hot_prior.sample().observation_space.ndim, 2)

        default_prior = TigerPrior(EncodeType.DEFAULT)
        self.assertEqual(default_prior.sample().observation_space.ndim, 1)

    def test_observation_prob(self) -> None:
        """ tests the observation probability of samples """

        if random.choice([True, False]):
            encoding = EncodeType.DEFAULT
        else:
            encoding = EncodeType.ONE_HOT

        tiger = TigerPrior(encoding).sample()
        assert isinstance(tiger, Tiger)

        obs_probs = tiger._correct_obs_probs  # pylint: disable=protected-access
        self.assertTrue(0 <= obs_probs[0] <= 1)
        self.assertTrue(0 <= obs_probs[1] <= 1)


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

        print(sample_gridworld_1.slow_cells)
        print(sample_gridworld_2.slow_cells)

        self.assertTrue(sample_gridworld_1.slow_cells, "may **rarely** be empty")
        self.assertTrue(sample_gridworld_2.slow_cells, "may **rarely** be empty")
        self.assertTrue(sample_gridworld_1.slow_cells != sample_gridworld_2.slow_cells, "may **rarely** be true")
