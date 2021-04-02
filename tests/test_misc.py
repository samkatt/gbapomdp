"""tests :mod:`general_bayes_adaptive_pomdps.misc`"""

import random
import unittest

import numpy as np

from general_bayes_adaptive_pomdps.misc import DiscreteSpace, set_random_seed


class TestSpaces(unittest.TestCase):
    """ tests the discrete space """

    def test_num_elements(self):
        """ tests whether the number of elements is as expected """

        space = DiscreteSpace([3, 2, 3])

        self.assertEqual(space.n, space.num_elements)
        self.assertEqual(space.n, 18)

    def test_num_dimensions(self):
        """" tests whether it correctly returns the number of dimensions """

        space = DiscreteSpace([3] * 8)

        self.assertEqual(space.ndim, 8)

    def test_sample(self):
        """ tests sampling """

        space = DiscreteSpace([5, 2])

        self.assertTrue(space.contains(space.sample()))

    def test_contain(self):
        """ tests contains """

        space = DiscreteSpace([2, 3])

        self.assertTrue(space.contains(np.array([0, 0])))
        self.assertFalse(space.contains(np.array([-1, 0])))
        self.assertFalse(space.contains(np.array([0, 3])))

    def test_index_of(self):
        """ tests getting index of things """

        single_dim_space = DiscreteSpace([5])
        self.assertEqual(single_dim_space.index_of(np.array([0])), 0)
        self.assertEqual(single_dim_space.index_of(np.array([2])), 2)
        self.assertEqual(single_dim_space.index_of(np.array([4])), 4)

        multi_dim_space = DiscreteSpace([3, 2, 5])
        self.assertEqual(multi_dim_space.index_of(np.array([0, 0, 0])), 0)
        self.assertEqual(multi_dim_space.index_of(np.array([2, 0, 0])), 2)
        self.assertEqual(multi_dim_space.index_of(np.array([0, 1, 0])), 3)
        self.assertEqual(multi_dim_space.index_of(np.array([0, 0, 3])), 18)
        self.assertEqual(multi_dim_space.index_of(np.array([2, 0, 3])), 20)
        self.assertEqual(multi_dim_space.index_of(np.array([2, 1, 2])), 17)

        edge_case_space = DiscreteSpace([2, 1, 3])
        self.assertEqual(edge_case_space.index_of(np.array([0, 0, 2])), 4)


class TestRandomSeed(unittest.TestCase):
    def test_default_behaviour(self) -> None:
        """ regular sampling """

        random_sample = random.uniform(0, 1)
        self.assertNotAlmostEqual(random_sample, random.uniform(0, 1))

        random_np_sample = np.random.uniform(0, 1)
        self.assertNotAlmostEqual(random_np_sample, np.random.uniform(0, 1))

    def test_setting_seed(self) -> None:
        """ tests whether setting the seed will result in repetitive behaviour """

        seed = random.randint(0, 1000)
        set_random_seed(seed)

        random_sample = random.uniform(0, 1)
        random_np_sample = np.random.uniform(0, 1)

        set_random_seed(seed)

        self.assertAlmostEqual(random_sample, random.uniform(0, 1))
        self.assertAlmostEqual(random_np_sample, np.random.uniform(0, 1))


if __name__ == "__main__":
    unittest.main()
