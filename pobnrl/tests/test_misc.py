""" runs tests on misc functionality

DiscreteSpace

"""

import unittest

import numpy as np

from misc import DiscreteSpace
from agents.neural_networks.misc import ReplayBuffer


class TestDiscreteSpace(unittest.TestCase):
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


class TestReplayBuffer(unittest.TestCase):
    """ Tests some functionality of the replay buffer """

    def test_capacity(self):
        """ tests the capacity property of the replay buffer """

        replay_buffer = ReplayBuffer()

        self.assertEqual(replay_buffer.capacity, 5000)

        replay_buffer.store_2((), False)
        self.assertEqual(replay_buffer.capacity, 5000)

        replay_buffer.store_2((), False)
        self.assertEqual(replay_buffer.capacity, 5000)

        replay_buffer.store_2((), True)
        self.assertEqual(replay_buffer.capacity, 5000)

    def test_size(self):
        """ tests the size property of replay buffer """

        replay_buffer = ReplayBuffer()

        self.assertEqual(replay_buffer.size_2, 1)

        replay_buffer.store_2((), False)
        self.assertEqual(replay_buffer.size_2, 1)

        replay_buffer.store_2((), False)
        self.assertEqual(replay_buffer.size_2, 1)

        replay_buffer.store_2((), True)
        self.assertEqual(replay_buffer.size_2, 2)

        replay_buffer.store_2((), False)
        self.assertEqual(replay_buffer.size_2, 2)

        replay_buffer.store_2((), True)
        replay_buffer.store_2((), True)
        self.assertEqual(replay_buffer.size_2, 4)

        replay_buffer.store_2((), True)
        self.assertEqual(replay_buffer.size_2, 5)


if __name__ == '__main__':
    unittest.main()
