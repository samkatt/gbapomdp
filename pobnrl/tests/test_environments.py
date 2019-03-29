""" runs tests on the environments """

import unittest

import numpy as np

from environments import gridworld, tiger, collision_avoidance


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

        obs = [self.env.reset() for _ in range(0, 10)]
        for observation in obs:
            np.testing.assert_array_equal(observation, [0, 0])

    def test_step(self):
        """ tests some basic dynamics """

        state = self.env.state

        obs = []
        # tests effect of listening
        for _ in range(0, 50):
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

            np.testing.assert_array_equal(obs, [0, 0])
            self.assertEqual(rew, 10)
            self.assertTrue(term)

        # test opening correct door
        for _ in range(0, 5):
            self.env.reset()
            open_wrong_door = 1 - self.env.state  # implementation knowledge
            obs, rew, term = self.env.step(open_wrong_door)

            np.testing.assert_array_equal(obs, [0, 0])
            self.assertEqual(rew, -100)
            self.assertTrue(term)

    def test_space(self):
        """ tests the size of the spaces """
        action_space = self.env.action_space
        observation_space = self.env.observation_space

        np.testing.assert_array_equal(action_space.dimensions, [3])
        np.testing.assert_array_equal(observation_space.dimensions, [2, 2])

        self.assertEqual(action_space.n, 3)
        self.assertEqual(observation_space.n, 4)


class TestGridWorld(unittest.TestCase):
    """ Tests for GridWorld environment """

    def test_reset(self):
        """ Tests Gridworld.reset() """
        env = gridworld.GridWorld(3, False)
        observation = env.reset()

        np.testing.assert_array_equal(env.state[0], [0, 0])
        self.assertIn(observation[2:].astype(int).tolist(), [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])

    def test_step(self):
        """ Tests Gridworld.step() """

        env = gridworld.GridWorld(3, False)
        env.reset()

        goal = env.state[1]

        # left and low keeps in place
        actions = [env.WEST, env.SOUTH]
        for _ in range(10):
            _, rew, term = env.step(np.random.choice(actions))

            np.testing.assert_array_equal(env.state[0], [0, 0])
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

        env.state = [np.array([2, 2]), (2, 2)]
        _, rew, term = env.step(np.random.randint(0, 4))

        self.assertEqual(rew, 1)
        self.assertTrue(term)

    def test_space(self):
        """ Tests Gridworld.spaces """
        env = gridworld.GridWorld(5, False)
        action_space = env.action_space
        observation_space = env.observation_space

        self.assertEqual(action_space.n, 4)
        self.assertTupleEqual(action_space.shape, (1,))

        self.assertEqual(observation_space.n, 25 * pow(2, len(env.goals)))
        self.assertTupleEqual(observation_space.shape, (2 + len(env.goals),))

    def test_utils(self):
        """ Tests misc functionality in Gridworld

        * Gridworld.generate_observation()
        * Gridworld.bound_in_grid()
        * Gridworld.sample_goal()
        * Gridworld.goals
        * Gridworld.space functionality
        * Gridworld.size

        """

        env = gridworld.GridWorld(3, False)

        # test bound_in_grid
        np.testing.assert_array_equal(
            env.bound_in_grid(np.array([2, 2])), [2, 2]
        )

        np.testing.assert_array_equal(
            env.bound_in_grid(np.array([3, 2])), [2, 2]
        )

        np.testing.assert_array_equal(
            env.bound_in_grid(np.array([-1, 3])), [0, 2]
        )

        # test sample_goal
        self.assertIn(env.sample_goal(), [(1, 2), (2, 1), (2, 2)])

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

        # generate_observation
        np.testing.assert_array_almost_equal(
            env.obs_mult, [.05, .05, .8, .05, .05]
        )
        env.obs_mult = np.array([0, 0, 1, 0, 0])

        observation = env.generate_observation([0, 0], (2, 2))
        np.testing.assert_array_equal(observation[2:], [0, 0, 1])
        np.testing.assert_array_equal(observation[:2], [0, 0])

        observation = env.generate_observation([1, 1], (2, 1))
        np.testing.assert_array_equal(observation[2:], [0, 1, 0])
        np.testing.assert_array_equal(observation[:2], [1, 1])

        env.obs_mult = np.array([0, .5, 0, .5, 0])
        observation = env.generate_observation([0, 0], (2, 2))
        self.assertIn(observation[0], [0, 1])
        self.assertIn(observation[1], [0, 1])

        observation = env.generate_observation([1, 1], (2, 2))
        self.assertIn(observation[0], [0, 2])
        self.assertIn(observation[1], [0, 2])

        env.obs_mult = np.array([1, 0, 0, 0, 0])
        observation = env.generate_observation([1, 2], (2, 2))
        self.assertEqual(observation[0], 0)
        self.assertEqual(observation[1], 0)


class TestCollisionAvoidance(unittest.TestCase):
    """ Tests Collision Avoidance class """

    def test_reset(self):
        """ tests CollisionAvoidance.reset """

        env = collision_avoidance.CollisionAvoidance(3, False)
        env.reset()

        self.assertEqual(env.state['obstacle'], 1)
        self.assertEqual(env.state['agent_x'], 2)
        self.assertEqual(env.state['agent_y'], 1)

    def test_step(self):
        """ tests CollisionAvoidance.step """

        env = collision_avoidance.CollisionAvoidance(7, False)

        env.reset()
        _, rew, term = env.step(1)
        self.assertEqual(env.state['agent_x'], 5)
        self.assertEqual(env.state['agent_y'], 3)
        self.assertFalse(term)
        self.assertEqual(rew, 0)

        env.reset()
        _, rew, term = env.step(0)
        self.assertEqual(env.state['agent_x'], 5)
        self.assertEqual(env.state['agent_y'], 2)
        self.assertFalse(term)
        self.assertEqual(rew, -1)

        env.reset()
        _, rew, term = env.step(2)
        self.assertEqual(env.state['agent_x'], 5)
        self.assertEqual(env.state['agent_y'], 4)
        self.assertFalse(term)
        self.assertEqual(rew, -1)

        _, rew, term = env.step(2)
        self.assertEqual(env.state['agent_x'], 4)
        self.assertEqual(env.state['agent_y'], 5)
        self.assertFalse(term)
        self.assertEqual(rew, -1)

        env.step(1)
        env.step(1)
        env.step(1)
        _, rew, term = env.step(1)

        self.assertEqual(env.state['agent_x'], 0)
        self.assertEqual(env.state['agent_y'], 5)
        self.assertTrue(term)

        should_be_rew = -1000 if env.state['obstacle'] == 5 else 0
        self.assertEqual(rew, should_be_rew)

    def test_utils(self):
        """ Tests misc functionality in Collision Avoidance

        * CollisionAvoidance.generate_observation()
        * CollisionAvoidance.bound_in_grid()
        * CollisionAvoidance.action_sace
        * CollisionAvoidance.observation_space
        * CollisionAvoidance.size

        """

        env = collision_avoidance.CollisionAvoidance(7, False)

        # action_space
        self.assertEqual(env.action_space.n, 3)
        self.assertTupleEqual(env.action_space.shape, (1,))
        np.testing.assert_array_equal(env.action_space.dimensions, [3])

        # observation_space
        self.assertEqual(env.observation_space.n, 7 * 7 * 7)
        self.assertTupleEqual(env.observation_space.shape, (3,))
        np.testing.assert_array_equal(
            env.observation_space.dimensions, [7, 7, 7]
        )

        # bound_in_grid
        self.assertEqual(env.bound_in_grid(8), 6)
        self.assertEqual(env.bound_in_grid(2), 2)
        self.assertEqual(env.bound_in_grid(-2), 0)

        # size
        self.assertEqual(env.size, 7)

        obs = env.generate_observation()
        self.assertTupleEqual(obs.shape, (3,))
        np.testing.assert_array_equal(obs[:2], [6, 3])
        self.assertIn(obs[2], list(range(6)))

        obs = env.generate_observation(
            {'agent_x': 2, 'agent_y': 4, 'obstacle': 3}
        )
        self.assertTupleEqual(obs.shape, (3,))
        np.testing.assert_array_equal(obs[:2], [2, 4])
        self.assertIn(obs[2], list(range(6)))


if __name__ == '__main__':
    unittest.main()
