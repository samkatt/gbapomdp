""" runs tests on misc functionality """

import unittest
import random

import numpy as np

from misc import sample_n_unique, sample_n


class TestSampling(unittest.TestCase):
    """ sampling tests class

    Tests sampling

    """

    def test_unique_sampling(self):
        """ tests unique sampling """

        def three_sampler():
            return 3

        self.assertEqual(sample_n_unique(three_sampler, 1), [3])

        def random_sampler():
            return random.randint(0, 5)
        assert (np.array(sample_n_unique(random_sampler, 5)) < 6).all()

    def test_sampling(self):
        """ tests unique sampling """

        def three_sampler():
            return 3

        self.assertEqual(sample_n(three_sampler, 10), [3] * 10)

        def random_sampler():
            return random.randint(0, 5)
        assert (np.array(sample_n(random_sampler, 5)) < 6).all()


if __name__ == '__main__':
    unittest.main()
