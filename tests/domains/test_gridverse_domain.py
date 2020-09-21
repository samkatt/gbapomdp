""" runs tests on the grid verse domain """
import unittest
from functools import reduce
from operator import mul

import numpy as np
import po_nrl.domains.gridverse_domain as gverse
from gym_gridverse.envs.factory import gym_minigrid_from_descr
from po_nrl.domains import GridverseDomain
from po_nrl.misc import DiscreteSpace


class TestGridverseDomain(unittest.TestCase):
    """Tests `GridverseDomain`"""

    def setUp(self):

        self.env = GridverseDomain(
            gym_minigrid_from_descr('MiniGrid-Empty-5x5-v0')
        )

    def test_state_or_observation_conversions(self):
        """Tests conversions between `flatten_..` and `reshape...`"""

        grid_shape = (7, 7)

        s = self.env.sample_start_state()
        o = self.env.reset()

        # initial values are as expected
        self.assertTupleEqual(
            s.shape, (reduce(mul, grid_shape) + 3,),
        )
        self.assertTupleEqual(
            o.shape, (reduce(mul, grid_shape),),
        )

        reshaped = gverse.reshape_state_or_observation(s, 7, 7)
        # conversion produces similar values
        self.assertTupleEqual(reshaped['grid'].shape, grid_shape)
        # initial position should be (0,0) facing east (= 2)
        np.testing.assert_array_equal(reshaped['agent'], [1, 1, 2])

    def test_sample_start_state(self):
        """Basic property tests on initial state"""

        s = gverse.reshape_state_or_observation(
            self.env.sample_start_state(), 7, 7
        )

        # position of goal
        self.assertEqual(s['grid'][-2, -2], 4)
        # initial agent position should be (1,1) facing east (= 2)
        np.testing.assert_array_equal(s['agent'][:3], [1, 1, 2])

    def test_spaces(self):
        """tests the action/state/observation space of grid verse"""
        self.assertEqual(self.env.action_space.n, 6)

    def test_functions_do_not_crash(self):
        """calls all untested functions to make sure they at least run"""

        s = self.env.sample_start_state()
        self.assertRaises(NotImplementedError, self.env.simulation_step, s, 1)

        self.env.reset()
        self.env.step(0)

        # TODO: actually test properties of
        self.assertIsInstance(self.env.observation_space, DiscreteSpace)
        self.assertIsInstance(self.env.state_space, DiscreteSpace)

        self.assertEqual(self.env.reward(s, 3, s), 0)
        self.assertFalse(self.env.terminal(s, 3, s))


if __name__ == '__main__':
    unittest.main()
