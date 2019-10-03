""" runs tests on misc functionality

* DiscreteSpace
* replay buffer
* softmax sampling
* random seed

"""

import unittest

import numpy as np
import random

from po_nrl.agents.neural_networks.misc import ReplayBuffer
from po_nrl.environments import ActionSpace
from po_nrl.misc import DiscreteSpace, set_random_seed


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


class TestRandomSeed(unittest.TestCase):
    """ tests setting the random seed """

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


if __name__ == '__main__':
    unittest.main()
