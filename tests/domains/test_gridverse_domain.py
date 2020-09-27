""" runs tests on the grid verse domain """
import unittest
from functools import reduce
from operator import mul

import numpy as np
from gym_gridverse.envs.factory import gym_minigrid_from_descr
from po_nrl.domains import GridverseDomain


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
            s.shape, (reduce(mul, grid_shape) + 7 + 7 + 4,),
        )
        self.assertTupleEqual(
            o.shape, (reduce(mul, grid_shape),),
        )

        np.testing.assert_array_equal(
            s[7 * 7:],
            np.concatenate(
                [[0, 1], np.zeros(5), [0, 1], np.zeros(5), [0, 0, 1, 0]]
            ),
        )

        # pylint: disable=protected-access
        grid, pos, orient = self.env._state_encoding.decode(s)

        # conversion produces similar values
        self.assertTupleEqual(grid.shape, grid_shape)

        # initial position should be (1,1) facing east (= 2)
        np.testing.assert_array_equal(pos, [1, 1])
        self.assertEqual(orient.value, 2)

    def test_sample_start_state(self):
        """Basic property tests on initial state"""

        # pylint: disable=protected-access
        grid, pos, orient = self.env._state_encoding.decode(
            self.env.sample_start_state()
        )

        # position of goal
        self.assertEqual(grid[-2, -2], 4)
        # initial agent position should be (1,1) facing east (= 2)
        np.testing.assert_array_equal(pos, [1, 1])
        self.assertEqual(orient.value, 2)

    def test_action_space(self):
        """tests the action space of grid verse"""
        # action
        self.assertEqual(self.env.action_space.n, 6)

    def test_state_space(self):
        """tests the state space of grid verse"""

        num_grid_dim = self.env.h * self.env.w
        num_pos_dim = self.env.h + self.env.w
        num_orientation_dim = 4

        self.assertEqual(
            self.env.state_space.ndim,
            num_grid_dim + num_pos_dim + num_orientation_dim,
        )

        state_space_size = np.ones(self.env.state_space.ndim, dtype=int) * 5

        # one-hot encoding of position and orientation
        state_space_size[num_grid_dim:] = 2

        np.testing.assert_array_equal(
            self.env.state_space.size, state_space_size  # type: ignore
        )

    def test_obs_space(self):
        """tests the obs space of grid verse"""

        self.assertEqual(
            self.env.observation_space.ndim, self.env.obs_h * self.env.obs_w
        )

        obs_space_size = np.ones(self.env.observation_space.ndim, dtype=int) * 5

        np.testing.assert_array_equal(
            self.env.observation_space.size, obs_space_size  # type: ignore
        )

    def test_functions_do_not_crash(self):
        """calls all untested functions to make sure they at least run"""

        s = self.env.sample_start_state()
        self.assertRaises(
            NotImplementedError, self.env.simulation_step, s, action=1
        )

        self.env.reset()
        self.env.step(0)

        self.assertEqual(self.env.reward(s, 3, s), 0)
        self.assertFalse(self.env.terminal(s, 3, s))


if __name__ == '__main__':
    unittest.main()
