""" runs tests on planners (PO-UCT)"""

import math
import random
import unittest

import numpy as np

from agents.planning.pouct import POUCT, TreeNode
from domains import Tiger


class TestPOUCTNode(unittest.TestCase):
    """ class to test the TreeNode of the POUCT """

    def setUp(self):
        """ creates a node for random # childs and depth """
        self.depth = random.randint(0, 4)
        self.num_children = random.randint(2, 4)

        self.node = TreeNode(self.num_children, self.depth)

        self.extended_child = self.node.child(0, 5)

    def test_creation(self):
        """ tests creating the nodes """

        self.assertEqual(self.node.depth, self.depth)
        self.assertEqual(self.node.num_children, self.num_children)

        np.testing.assert_array_equal(
            self.node.avg_values, [0] * self.num_children
        )

        np.testing.assert_array_equal(
            self.node.children_visits, [0] * self.num_children
        )

    def test_adding_first_value(self):
        """ tests adding the first value to a child node """
        self.node.update_value(0, 10)

        self.assertEqual(self.node.avg_values[0], 10)
        self.assertEqual(self.node.children_visits[0], 1)

        self.assertEqual(self.node.avg_values[1], 0)
        self.assertEqual(self.node.children_visits[1], 0)

        # cannot increment value outside of range
        self.assertRaises(IndexError, self.node.update_value, 10, 0)

    def test_adding_multiple_values(self):
        """ tests what happens if the same child updates mulitple values """
        chosen_child = random.randint(0, self.num_children - 1)

        self.node.update_value(chosen_child, 5)
        self.node.update_value(chosen_child, 10)

        self.assertAlmostEqual(self.node.avg_values[chosen_child], 7.5)
        self.assertEqual(self.node.children_visits[chosen_child], 2)

        self.node.update_value(chosen_child, 30)

        self.assertAlmostEqual(self.node.avg_values[chosen_child], 15)
        self.assertEqual(self.node.children_visits[chosen_child], 3)

        other_child = 0 if chosen_child != 0 else 1

        self.assertEqual(self.node.avg_values[other_child], 0)
        self.assertEqual(self.node.children_visits[other_child], 0)

    def test_children(self):
        """ tests whether accessing earlier created childs are working """

        self.assertEqual(self.extended_child.depth, self.depth + 1)
        self.assertEqual(self.extended_child.num_children, self.num_children)

        np.testing.assert_array_equal(
            self.extended_child.avg_values, [0] * self.num_children
        )

        np.testing.assert_array_equal(
            self.extended_child.children_visits, [0] * self.num_children
        )

        self.assertTrue(self.extended_child is self.node.child(0, 5))


class TestPOUCT(unittest.TestCase):
    """ class to test POUCT """

    def test_ucb(self):
        """ tests the UCB values """

        # some error checking

        functional_table = np.array([[float('inf'), 2]])

        self.assertRaises(
            AssertionError,
            POUCT.ucb, np.array([1]), np.array([1, 1]), functional_table
        )

        self.assertRaises(
            AssertionError,
            POUCT.ucb, np.array([1, 1]), np.array([1]), functional_table
        )

        self.assertRaises(
            AssertionError,
            POUCT.ucb, np.array([1]), np.array([-1]), functional_table
        )

        bad_table = np.array([[]])
        self.assertRaises(
            IndexError,
            POUCT.ucb, np.array([1]), np.array([1]), bad_table
        )

        q_vals = np.array([1, 2])
        visits = np.array([0, 1]).astype(int)

        zero_bonus_table = np.zeros((10, 10))
        np.testing.assert_array_equal(
            POUCT.ucb(q_vals, visits, zero_bonus_table), [1, 2]
        )

        ones_bonus_table = np.ones((10, 10))
        np.testing.assert_array_equal(
            POUCT.ucb(q_vals, visits, ones_bonus_table), [2, 3]
        )

        incrementing_table = np.tile(np.arange(10), (10, 1))
        np.testing.assert_array_equal(
            POUCT.ucb(q_vals, visits, incrementing_table), [1, 3]
        )

        infinite_table = np.full((10, 10), float('inf'))
        np.testing.assert_array_equal(
            POUCT.ucb(q_vals, visits, infinite_table), [float('inf')] * 2
        )

        q_vals = np.array([.5, -.2])
        visits = np.array([5, 2])

        np.testing.assert_array_equal(
            POUCT.ucb(q_vals, visits, infinite_table), [float('inf')] * 2
        )
        np.testing.assert_array_equal(
            POUCT.ucb(q_vals, visits, incrementing_table), [5.5, 1.8]
        )
        np.testing.assert_array_equal(
            POUCT.ucb(q_vals, visits, zero_bonus_table), q_vals
        )
        np.testing.assert_array_equal(
            POUCT.ucb(q_vals, visits, ones_bonus_table), q_vals + 1
        )

    def test_ucb_table(self):
        """ tests the values in a ucb table """

        domain = Tiger()

        table = POUCT(  # pylint: disable=protected-access
            domain,
            num_sims=5,
            exploration_constant=1
        )._ucb_table

        self.assertTupleEqual(table.shape, (5, 5))
        self.assertTrue(np.all(table[1:, 0] == float('inf')))

        self.assertAlmostEqual(table[1, 1], math.sqrt(math.log(2) / 1))
        self.assertAlmostEqual(table[1, 2], math.sqrt(math.log(2) / 2))
        self.assertAlmostEqual(table[3, 2], math.sqrt(math.log(4) / 2))
        self.assertAlmostEqual(table[3, 4], math.sqrt(math.log(4) / 4))

        table = POUCT(  # pylint: disable=protected-access
            domain,
            num_sims=5,
            exploration_constant=50.3
        )._ucb_table

        self.assertAlmostEqual(table[1, 1], 50.3 * math.sqrt(math.log(2) / 1))
        self.assertAlmostEqual(table[1, 2], 50.3 * math.sqrt(math.log(2) / 2))
        self.assertAlmostEqual(table[3, 2], 50.3 * math.sqrt(math.log(4) / 2))
        self.assertAlmostEqual(table[3, 4], 50.3 * math.sqrt(math.log(4) / 4))


if __name__ == '__main__':
    unittest.main()
