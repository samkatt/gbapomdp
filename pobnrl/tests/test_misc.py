""" runs tests on misc functionality

DiscreteSpace

"""

import unittest

import numpy as np

from agents.neural_networks.misc import ReplayBuffer
from agents.neural_networks.neural_pomdps import DynamicsModel as DM
from environments import ActionSpace
from misc import DiscreteSpace


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

    def test_one_hot(self):  # pylint: disable=no-self-use
        """ tests 1-hot encoding of actions """

        action_space = ActionSpace(5)

        np.testing.assert_array_equal(action_space.one_hot(0), [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(action_space.one_hot(4), [0, 0, 0, 0, 1])
        np.testing.assert_array_equal(action_space.one_hot(3), [0, 0, 0, 1, 0])


class TestReplayBuffer(unittest.TestCase):
    """ Tests some functionality of the replay buffer """

    def test_capacity(self):
        """ tests the capacity property of the replay buffer """

        replay_buffer = ReplayBuffer()

        self.assertEqual(replay_buffer.capacity, 5000)

        replay_buffer.store((), False)
        self.assertEqual(replay_buffer.capacity, 5000)

        replay_buffer.store((), False)
        self.assertEqual(replay_buffer.capacity, 5000)

        replay_buffer.store((), True)
        self.assertEqual(replay_buffer.capacity, 5000)

    def test_size(self):
        """ tests the size property of replay buffer """

        replay_buffer = ReplayBuffer()

        self.assertEqual(replay_buffer.size, 1)

        replay_buffer.store((), False)
        self.assertEqual(replay_buffer.size, 1)

        replay_buffer.store((), False)
        self.assertEqual(replay_buffer.size, 1)

        replay_buffer.store((), True)
        self.assertEqual(replay_buffer.size, 2)

        replay_buffer.store((), False)
        self.assertEqual(replay_buffer.size, 2)

        replay_buffer.store((), True)
        replay_buffer.store((), True)
        self.assertEqual(replay_buffer.size, 4)

        replay_buffer.store((), True)
        self.assertEqual(replay_buffer.size, 5)


class TestSoftmax(unittest.TestCase):
    """ tests the softmax sampling method """

    def test_simple(self):
        """ some super easy stuff """

        self.assertEqual(DM.softmax_sample(np.array([1])), 0)
        self.assertEqual(DM.softmax_sample(np.array([25])), 0)
        self.assertEqual(DM.softmax_sample(np.array([0, 5])), 1)
        self.assertEqual(DM.softmax_sample(np.array([100, 0])), 0)

    def test_negative(self):
        """ tests with negative numbers """

        self.assertEqual(DM.softmax_sample(np.array([-10])), 0)
        self.assertEqual(DM.softmax_sample(np.array([-10, 1])), 1)
        self.assertEqual(DM.softmax_sample(np.array([-10, -1])), 1)


if __name__ == '__main__':
    unittest.main()
