""" runs tests on the environments """

import unittest

import numpy as np

from environments import gridworld
from environments import tiger


class TestTiger(unittest.TestCase):
    """ tests functionality of the tiger environment """

    @classmethod
    def setUpClass(cls):
        """ creates a tiger member """
        cls.env = tiger.Tiger(False)
        cls.env.reset()

    def test_reset(self):
        """ tests that start state is 0 or 1 """

        self.assertIn(self.env.state, [0, 1])

        states = [self.env.sample_start_state() for _ in range(0, 20)]

        for state in states:
            self.assertIn(state, [0, 1], 'state should be either 0 or 1')

        self.assertIn(0, states, 'there should be at least one of this state')
        self.assertIn(1, states, 'there should be at least one of this state')

        obs = [self.env.reset().tolist() for _ in range(0, 10)]
        for observation in obs:
            self.assertListEqual(observation, [0, 0])

    def test_step(self):
        """ tests some basic dynamics """

        state = self.env.state

        obs = []
        # tests effect of listening
        for _ in range(0, 30):
            observation, rew, term = self.env.step(self.env.LISTEN)
            self.assertEqual(state, self.env.state)
            self.assertIn(observation.tolist(), [[0, 0], [0, 1], [1, 0]])
            self.assertFalse(term)
            self.assertEqual(rew, -1.0)

            obs.append(observation.tolist())

        # tests stochasticity of observations when listening
        self.assertNotIn([0, 0], obs)
        self.assertIn([0, 1], obs)
        self.assertIn([1, 0], obs)

        # test opening correct door
        for _ in range(0, 5):
            self.env.reset()
            open_correct_door = self.env.state  # implementation knowledge
            obs, rew, term = self.env.step(open_correct_door)

            self.assertListEqual(obs.tolist(), [0, 0])
            self.assertEqual(rew, 10)
            self.assertTrue(term)

        # test opening correct door
        for _ in range(0, 5):
            self.env.reset()
            open_wrong_door = 1 - self.env.state  # implementation knowledge
            obs, rew, term = self.env.step(open_wrong_door)

            self.assertListEqual(obs.tolist(), [0, 0])
            self.assertEqual(rew, -100)
            self.assertTrue(term)

    def test_space(self):
        """ tests the size of the spaces """
        spaces = self.env.spaces()

        self.assertListEqual(spaces['A'].dimensions.tolist(), [3])
        self.assertListEqual(spaces['O'].dimensions.tolist(), [2, 2])

        self.assertEqual(spaces['A'].n, 3)
        self.assertEqual(spaces['O'].n, 4)


class TestGridWorld(unittest.TestCase):
    """ Tests for GridWorld environment """

    def test_reset(self):
        """ Tests Gridworld.reset() """
        env = gridworld.GridWorld(3, False)
        observation = env.reset()

        self.assertListEqual(env.state[0].tolist(), [0, 0])
        self.assertIn(observation[2:].astype(int).tolist(), [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])

    def test_step(self):
        """ Tests Gridworld.step()

        TODO: nyi

        """

        env = gridworld.GridWorld(3, False)
        env.reset()

        goal = env.state[1]

        # left and low keeps in place
        actions = [env.WEST, env.SOUTH]
        for _ in range(10):
            _, rew, term = env.step(np.random.choice(actions))

            self.assertListEqual(env.state[0].tolist(), [0, 0])
            self.assertEqual(rew, 0)
            self.assertFalse(term)
            self.assertEqual(env.state[1], goal)

        # test step up
        _, rew, term = env.step(env.NORTH)

        self.assertIn(env.state[0].tolist(), [[0, 0], [0, 1]])
        self.assertFalse(term)
        self.assertEqual(rew, 0)
        self.assertEqual(env.state[1], goal)

        # test step east
        env.reset()
        goal = env.state[1]
        _, rew, term = env.step(env.EAST)

        self.assertIn(env.state[0].tolist(), [[0, 0], [1, 0]])
        self.assertFalse(term)
        self.assertEqual(rew, 0)
        self.assertEqual(env.state[1], goal)

    def test_space(self):
        """ Tests Gridworld.spaces() """
        env = gridworld.GridWorld(5, False)
        spaces = env.spaces()

        self.assertEqual(spaces['A'].n, 4)
        self.assertTupleEqual(spaces['A'].shape, (1,))

        self.assertEqual(spaces['O'].n, 25 * pow(2, len(env.goals)))
        self.assertTupleEqual(spaces['O'].shape, (2 + len(env.goals),))

    def test_utils(self):
        """ Tests misc functionality in Gridworld

        * Gridworld.generateObservation()
        * Gridworld.bound_in_grid()
        * Gridworld.sample_goal()
        * Gridworld.goals

        """

        env = gridworld.GridWorld(3, False)

        # test bound_in_grid
        self.assertListEqual(
            env.bound_in_grid(np.array([2, 2])).tolist(),
            [2, 2]
        )

        self.assertListEqual(
            env.bound_in_grid(np.array([3, 2])).tolist(),
            [2, 2]
        )

        self.assertListEqual(
            env.bound_in_grid(np.array([-1, 3])).tolist(),
            [0, 2]
        )

        # test sample_goal
        self.assertIn(env.sample_goal(), [(1, 2), (2, 1), (2, 2)])

        # generate_observation
        # TODO NYI

        # Gridworld.goals
        self.assertListEqual(env.goals, [(1, 2), (2, 1), (2, 2)])

        larger_env = gridworld.GridWorld(7, False)
        self.assertEqual(len(larger_env.goals), 10)

        # Gridworld.state
        test_state = [np.array([2, 2]), (2, 2)]
        env.state = test_state
        self.assertEqual(env.state, test_state)

        with self.assertRaises(AssertionError):
            env.state = [np.array([3, 2]), (2, 2)]

        with self.assertRaises(AssertionError):
            env.state = [np.array([0, 2]), (0, 2)]

        # Gridworld.size
        self.assertEqual(env.size, 3)
        self.assertEqual(larger_env.size, 7)


if __name__ == '__main__':
    unittest.main()
