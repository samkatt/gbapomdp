""" runs tests on the grid verse domain """

import unittest
from functools import partial, reduce
from operator import mul
from typing import Dict

import numpy as np
from gym_gridverse.actions import Actions as GverseAction
from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import MovingObstacle
from po_nrl.agents.neural_networks.neural_pomdps import DynamicsModel
from po_nrl.domains import GridverseDomain
from po_nrl.domains.gridverse_domain import ObservationModel as GverseObsModel
from po_nrl.domains.gridverse_domain import (StateEncoding,
                                             default_rollout_policy,
                                             straight_or_turn_policy)


class TestGridverseDomain(unittest.TestCase):
    """Tests `GridverseDomain`"""

    def setUp(self):

        self.env = GridverseDomain('one-hot-state', 'MiniGrid-Empty-5x5-v0')

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


class TestStateEncodings(unittest.TestCase):
    """tests implementations of `StateEncoding`"""

    # pylint: disable=protected-access
    def setUp(self):
        # NOTE: ugliest way of initiating possible
        self.obst_index = MovingObstacle.type_index  # pylint: disable=no-member

        self.env = GridverseDomain(
            "compact", "MiniGrid-Dynamic-Obstacles-Random-5x5-v0"
        )

        self.compact_encoding = StateEncoding.construct(
            "compact", self.env._state_encoding._rep,  # type: ignore
        )

        self.one_hot_orientation = StateEncoding.construct(
            "one-hot-orientation", self.env._state_encoding._rep  # type: ignore
        )

        self.one_hot_encoding = StateEncoding.construct(
            "one-hot-state", self.env._state_encoding._rep,  # type: ignore
        )

    def test_state_space(self):
        """tests `.state_space` implementation"""

        # compact
        compact_space = self.compact_encoding.state_space

        self.assertEqual(compact_space.ndim, 7 * 7 + 3)
        np.testing.assert_array_equal(
            compact_space.size, [self.obst_index + 1] * 7 * 7 + [7, 7, 4]
        )

        # one-hot orientation
        one_hot_orientation_space = self.one_hot_orientation.state_space

        self.assertEqual(one_hot_orientation_space.ndim, 7 * 7 + 2 + 4)
        np.testing.assert_array_equal(
            one_hot_orientation_space.size,
            [self.obst_index + 1] * 7 * 7 + [7, 7, 2, 2, 2, 2],
        )

        # one-hot
        one_hot_space = self.one_hot_encoding.state_space

        self.assertEqual(one_hot_space.ndim, 7 * 7 + 7 + 7 + 4)
        np.testing.assert_array_equal(
            one_hot_space.size,
            [self.obst_index + 1] * 7 * 7 + [2] * (7 + 7 + 4),
        )

    def test_codings(self):
        """tests encoding and decoding"""

        s = (
            self.env._gverse_env.functional_reset()  # pylint: disable=protected-access
        )

        for coding in [
            self.compact_encoding,
            self.one_hot_orientation,
            self.one_hot_encoding,
        ]:
            encoding = coding.encode(s)

            self.assertTrue(coding.state_space.contains(encoding))
            _, pos, orient = coding.decode(encoding)

            self.assertEqual(pos, s.agent.position)
            self.assertEqual(orient, s.agent.orientation)


class TestObservationModel(unittest.TestCase):
    """Tests `gridverse_domain.ObservationModel`"""

    def setUp(self):
        self.d = GridverseDomain()

        # pylint: disable=protected-access
        self.model = GverseObsModel(
            obs_size=7,
            encoding=self.d._state_encoding,
            max_item_index=self.d._gverse_env.state_space.max_grid_object_type
            + 1,
        )

    def test_sample(self):  # pylint: disable=no-self-use
        """basic `.sample()` functionality tests"""

        # initial state
        o = self.d.reset()
        s = self.d.state
        a = self.d.action_space.sample()

        np.testing.assert_equal(o, self.model.sample(s, a, s, num=1))

        s, a, next_s, o = self.d.sample_transition()
        np.testing.assert_equal(o, self.model.sample(s, a, next_s, num=1))

    def test_model(self):
        """tests basic properties of `.model()`"""

        s, a, next_s, o = self.d.sample_transition()
        O = self.model.model(s, a, next_s)

        np.testing.assert_array_equal(
            o, DynamicsModel.sample_from_model(O, num=1)
        )


class TestDefaultRolloutPolicy(unittest.TestCase):
    """tests `gridverse_domain.default_rollout_policy`"""

    def setUp(self):
        self.d = GridverseDomain()
        self.pol = partial(
            default_rollout_policy,
            encoding=self.d._state_encoding,  # pylint: disable=protected-access
        )

    def test_walls(self):
        """tests that policy does not go forward into walls"""
        s = self.d.sample_start_state()
        s[-1] = Orientation.N.value

        for _ in range(10):
            self.assertNotEqual(self.pol(s), GverseAction.MOVE_FORWARD)

    def test_basic(self):
        """tests whether all possible actions are sampled (and only those) and their frequency"""

        actions = [self.pol(self.d.sample_start_state()) for _ in range(25)]

        self.assertSetEqual(
            set(sorted(actions)),
            {
                GverseAction.TURN_LEFT.value,
                GverseAction.TURN_RIGHT.value,
                GverseAction.MOVE_FORWARD.value,
            },
        )

        freq: Dict[int, int] = {}
        for a in actions:
            freq[a] = freq.get(a, 0) + 1

        self.assertGreaterEqual(
            freq[GverseAction.MOVE_FORWARD.value],
            freq[GverseAction.TURN_LEFT.value],
        )

        self.assertGreaterEqual(
            freq[GverseAction.MOVE_FORWARD.value],
            freq[GverseAction.TURN_RIGHT.value],
        )


class TestStraightOrTurnRolloutPolicy(unittest.TestCase):
    """tests `gridverse_domain.straight_or_turn_policy`"""

    def setUp(self):
        self.d = GridverseDomain()
        self.pol = partial(
            straight_or_turn_policy,
            encoding=self.d._state_encoding,  # pylint: disable=protected-access
        )

    def test_walls(self):
        """tests that policy does not go forward into walls"""

        s = self.d.sample_start_state()
        s[-1] = Orientation.N.value

        actions = sorted({self.pol(s) for _ in range(25)})
        self.assertListEqual(
            actions,
            [GverseAction.TURN_LEFT.value, GverseAction.TURN_RIGHT.value],
        )

    def test_basic(self):
        """tests whether all possible actions are sampled (and only those) and their frequency"""

        actions = {self.pol(self.d.sample_start_state()) for _ in range(25)}
        self.assertSetEqual(actions, {GverseAction.MOVE_FORWARD.value})


if __name__ == '__main__':
    unittest.main()
