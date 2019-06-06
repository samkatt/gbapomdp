""" runs tests on the domains """

import unittest

import random

import numpy as np

from domains import gridworld, tiger, collision_avoidance, chain_domain


class TestTiger(unittest.TestCase):
    """ tests functionality of the tiger environment """

    def setUp(self):
        """ creates a tiger member """
        self.one_hot_env = tiger.Tiger(use_one_hot=True)
        self.one_hot_env.reset()

        self.env = tiger.Tiger(use_one_hot=False)
        self.env.reset()

    def test_reset(self):
        """ tests that start state is 0 or 1 """

        self.assertIn(self.one_hot_env.state, [0, 1])

        states = [self.one_hot_env.sample_start_state() for _ in range(0, 20)]

        for state in states:
            self.assertIn(state[0], [0, 1], 'state should be either 0 or 1')

        self.assertIn([0], states, 'there should be at least one of this state')
        self.assertIn([1], states, 'there should be at least one of this state')

        # one-hot observation
        obs = [self.one_hot_env.reset() for _ in range(0, 10)]
        for observation in obs:
            np.testing.assert_array_equal(observation, [1, 1])

        # regular observation
        obs = [self.env.reset() for _ in range(0, 10)]
        for observation in obs:
            np.testing.assert_array_equal(observation, [2])

    def test_step(self):
        """ tests some basic dynamics """

        state = self.one_hot_env.state

        obs = []
        # tests effect of listening
        for _ in range(0, 50):
            step = self.one_hot_env.step(self.one_hot_env.LISTEN)
            np.testing.assert_array_equal(state, self.one_hot_env.state)
            self.assertIn(step.observation.tolist(), [[0, 1], [1, 0]])
            self.assertFalse(step.terminal)
            self.assertEqual(step.reward, -1.0)

            obs.append(step.observation.tolist())

        # tests stochasticity of observations when listening
        self.assertNotIn([0, 0], obs)
        self.assertNotIn([1, 1], obs)
        self.assertIn([0, 1], obs)
        self.assertIn([1, 0], obs)

        # test opening correct door
        for _ in range(0, 5):
            self.one_hot_env.reset()
            open_correct_door = self.one_hot_env.state[0]  # implementation knowledge
            step = self.one_hot_env.step(open_correct_door)

            np.testing.assert_array_equal(step.observation, [1, 1])
            self.assertEqual(step.reward, 10)
            self.assertTrue(step.terminal)

        # test opening correct door
        for _ in range(0, 5):
            self.one_hot_env.reset()
            open_wrong_door = 1 - self.one_hot_env.state[0]  # implementation knowledge
            step = self.one_hot_env.step(open_wrong_door)

            np.testing.assert_array_equal(step.observation, [1, 1])
            self.assertEqual(step.reward, -100)
            self.assertTrue(step.terminal)

    def test_sample_start_state(self):
        """ tests sampling start states """

        start_states = [self.one_hot_env.sample_start_state() for _ in range(10)]

        self.assertIn([0], start_states)
        self.assertIn([1], start_states)

        for state in start_states:
            self.assertIn(state, [[0], [1]])

    def test_space(self):
        """ tests the size of the spaces """

        action_space = self.one_hot_env.action_space
        np.testing.assert_array_equal(action_space.size, [3])
        self.assertEqual(action_space.n, 3)

        one_hot_observation_space = self.one_hot_env.observation_space
        np.testing.assert_array_equal(one_hot_observation_space.size, [2, 2])
        self.assertEqual(one_hot_observation_space.n, 4)

        observation_space = self.env.observation_space
        np.testing.assert_array_equal(observation_space.size, [3])
        self.assertEqual(observation_space.n, 3)

    def test_observation_projection(self):
        """ tests tiger.obs2index """

        self.assertEqual(self.one_hot_env.obs2index(self.one_hot_env.reset()), 2)
        self.assertEqual(self.one_hot_env.obs2index(self.one_hot_env.reset()), 2)

        self.assertEqual(self.one_hot_env.obs2index(np.array([1, 0])), 0)
        self.assertEqual(self.one_hot_env.obs2index(np.array([0, 1])), 1)

        self.assertEqual(self.one_hot_env.obs2index(np.array([0, 0])), -1)
        self.assertEqual(self.one_hot_env.obs2index(np.array([1, 1])), 2)

    def test_observation_encoding(self):
        """ tests encoding of observation in Tiger """

        np.testing.assert_array_equal(self.one_hot_env.encode_observation(0), [1, 0])
        np.testing.assert_array_equal(self.one_hot_env.encode_observation(1), [0, 1])
        np.testing.assert_array_equal(self.one_hot_env.encode_observation(2), [1, 1])

        np.testing.assert_array_equal(self.env.encode_observation(0), [0])
        np.testing.assert_array_equal(self.env.encode_observation(1), [1])
        np.testing.assert_array_equal(self.env.encode_observation(2), [2])


class TestGridWorld(unittest.TestCase):
    """ Tests for GridWorld environment """

    def test_reset(self):
        """ Tests Gridworld.reset() """
        env = gridworld.GridWorld(3)
        observation = env.reset()

        np.testing.assert_array_equal(env.state[0], [0, 0])
        self.assertIn(observation[2:].astype(int).tolist(), [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])

    def test_sample_start_state(self):  # pylint: disable=no-self-use
        """ tests sampling start states """

        env = gridworld.GridWorld(5)

        start_states = [env.sample_start_state() for _ in range(10)]

        for state in start_states:
            np.testing.assert_array_equal(state[0], [0, 0])

    def test_step(self):
        """ Tests Gridworld.step() """

        env = gridworld.GridWorld(3)
        env.reset()

        goal = env.state[2]

        # left and low keeps in place
        actions = [env.WEST, env.SOUTH]
        for _ in range(10):
            step = env.step(np.random.choice(actions))

            np.testing.assert_array_equal(env.state[:2], [0, 0])
            self.assertEqual(step.reward, 0)
            self.assertFalse(step.terminal)
            self.assertEqual(env.state[2], goal)

        # test step up
        step = env.step(env.NORTH)

        self.assertIn(env.state[:2].tolist(), [[0, 0], [0, 1]])
        self.assertFalse(step.terminal)
        self.assertEqual(step.reward, 0)
        self.assertEqual(env.state[2], goal)

        # test step east
        env.reset()
        goal = env.state[2]
        step = env.step(env.EAST)

        self.assertIn(env.state[:2].tolist(), [[0, 0], [1, 0]])
        self.assertFalse(step.terminal)
        self.assertEqual(step.reward, 0)
        self.assertEqual(env.state[2], goal)

        random_goal = random.choice(env.goals)
        env.state = np.array([random_goal.x, random_goal.y, random_goal.index])
        step = env.step(np.random.randint(0, 4))

        self.assertEqual(step.reward, 1)
        self.assertTrue(step.terminal)

    def test_space(self):
        """ Tests Gridworld.spaces """
        env = gridworld.GridWorld(5)
        action_space = env.action_space
        observation_space = env.observation_space

        np.testing.assert_array_equal(action_space.size, [4])

        self.assertEqual(observation_space.n, 25 * pow(2, len(env.goals)))
        self.assertEqual(observation_space.ndim, 2 + len(env.goals))

    def test_utils(self):
        """ Tests misc functionality in Gridworld

        * Gridworld.generate_observation()
        * Gridworld.bound_in_grid()
        * Gridworld.sample_goal()
        * Gridworld.goals
        * Gridworld.space functionality
        * Gridworld.size

        """

        env = gridworld.GridWorld(3)

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

        list_of_goals = [
            gridworld.GridWorld.Goal(1, 2, 0),
            gridworld.GridWorld.Goal(2, 1, 1),
            gridworld.GridWorld.Goal(2, 2, 2),
        ]

        # test sample_goal
        self.assertIn(env.sample_goal(), list_of_goals)

        # Gridworld.goals
        self.assertListEqual(env.goals, list_of_goals)

        larger_env = gridworld.GridWorld(7)
        self.assertEqual(len(larger_env.goals), 10)

        # Gridworld.state
        random_goal = random.choice(env.goals)
        test_state = np.array([random_goal.x, random_goal.y, random_goal.index])
        env.state = test_state
        np.testing.assert_array_equal(env.state, test_state)

        with self.assertRaises(AssertionError):
            env.state = np.array([3, 2, random_goal.index])

        with self.assertRaises(AssertionError):
            env.state = np.array([0, 2, 5])

        # Gridworld.size
        self.assertEqual(env.size, 3)
        self.assertEqual(larger_env.size, 7)

        # generate_observation
        np.testing.assert_array_almost_equal(
            env.obs_mult, [.05, .05, .8, .05, .05]
        )
        env.obs_mult = np.array([0, 0, 1, 0, 0])

        observation = env.generate_observation(np.array([0, 0, 2]))
        np.testing.assert_array_equal(observation[2:], [0, 0, 1])
        np.testing.assert_array_equal(observation[:2], [0, 0])

        observation = env.generate_observation(np.array([1, 1, 1]))
        np.testing.assert_array_equal(observation[2:], [0, 1, 0])
        np.testing.assert_array_equal(observation[:2], [1, 1])

        env.obs_mult = np.array([0, .5, 0, .5, 0])
        observation = env.generate_observation(np.array([0, 0, 2]))
        self.assertIn(observation[0], [0, 1])
        self.assertIn(observation[1], [0, 1])

        observation = env.generate_observation(np.array([1, 1, 2]))
        self.assertIn(observation[0], [0, 2])
        self.assertIn(observation[1], [0, 2])

        env.obs_mult = np.array([1, 0, 0, 0, 0])
        observation = env.generate_observation(np.array([1, 2, 2]))
        self.assertEqual(observation[0], 0)
        self.assertEqual(observation[1], 0)

    def test_observation_projection(self):
        """ tests obs2index """

        env = gridworld.GridWorld(3)

        obs = np.array([0, 0, 1, 0, 0])

        for i in range(3):
            obs[0] = i
            self.assertEqual(env.obs2index(obs), i)

        obs[0] = 1
        obs[1] = 1
        self.assertEqual(env.obs2index(obs), 4)

        # increase goal index
        obs[2] = 0
        obs[3] = 1
        self.assertEqual(env.obs2index(obs), 13)

        obs[1] = 2
        self.assertEqual(env.obs2index(obs), 16)

        obs[-1] = 1
        self.assertRaises(AssertionError, env.obs2index, obs)

        obs = np.array([1, 2, 0, 0, 1])
        self.assertEqual(env.obs2index(obs), 25)


class TestCollisionAvoidance(unittest.TestCase):
    """ Tests Collision Avoidance class """

    def test_reset(self):
        """ tests CollisionAvoidance.reset """

        env = collision_avoidance.CollisionAvoidance(3)
        env.reset()

        self.assertEqual(env.state[0], 2)
        self.assertEqual(env.state[1], 1)
        self.assertEqual(env.state[2], 1)

    def test_sample_start_state(self):
        """ tests sampling start states """

        env = collision_avoidance.CollisionAvoidance(7)
        np.testing.assert_array_equal(env.sample_start_state(), [6, 3, 3])

    def test_step(self):
        """ tests CollisionAvoidance.step """

        env = collision_avoidance.CollisionAvoidance(7)

        env.reset()
        step = env.step(1)
        self.assertEqual(env.state[0], 5)
        self.assertEqual(env.state[1], 3)
        self.assertFalse(step.terminal)
        self.assertEqual(step.reward, 0)

        env.reset()
        step = env.step(0)
        self.assertEqual(env.state[0], 5)
        self.assertEqual(env.state[1], 2)
        self.assertFalse(step.terminal)
        self.assertEqual(step.reward, -1)

        env.reset()
        step = env.step(2)
        self.assertEqual(env.state[0], 5)
        self.assertEqual(env.state[1], 4)
        self.assertFalse(step.terminal)
        self.assertEqual(step.reward, -1)

        step = env.step(2)
        self.assertEqual(env.state[0], 4)
        self.assertEqual(env.state[1], 5)
        self.assertFalse(step.terminal)
        self.assertEqual(step.reward, -1)

        env.step(1)
        env.step(1)
        env.step(1)
        step = env.step(1)

        self.assertEqual(env.state[0], 0)
        self.assertEqual(env.state[1], 5)
        self.assertTrue(step.terminal)

        should_be_rew = -1000 if env.state[2] == 5 else 0
        self.assertEqual(step.reward, should_be_rew)

    def test_utils(self):
        """ Tests misc functionality in Collision Avoidance

        * CollisionAvoidance.generate_observation()
        * CollisionAvoidance.bound_in_grid()
        * CollisionAvoidance.action_sace
        * CollisionAvoidance.observation_space
        * CollisionAvoidance.size

        """

        env = collision_avoidance.CollisionAvoidance(7)

        # action_space
        self.assertEqual(env.action_space.n, 3)
        self.assertEqual(env.action_space.ndim, 1)
        np.testing.assert_array_equal(env.action_space.size, [3])

        # observation_space
        self.assertEqual(env.observation_space.n, 7 * 7 * 7)
        self.assertEqual(env.observation_space.ndim, 3)
        np.testing.assert_array_equal(
            env.observation_space.size, [7, 7, 7]
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
        self.assertIn(obs[2], list(range(7)))

        obs = env.generate_observation(np.array([2, 4, 3]))
        self.assertTupleEqual(obs.shape, (3,))
        np.testing.assert_array_equal(obs[:2], [2, 4])
        self.assertIn(obs[2], list(range(7)))

    def test_observation_projection(self):
        """ tests obs2index """

        env = collision_avoidance.CollisionAvoidance(domain_size=5)

        self.assertEqual(env.obs2index(np.array([0, 0, 0])), 0)
        self.assertEqual(env.obs2index(np.array([1, 0, 0])), 1)
        self.assertEqual(env.obs2index(np.array([0, 1, 0])), 5)
        self.assertEqual(env.obs2index(np.array([0, 0, 1])), 25)
        self.assertEqual(env.obs2index(np.array([2, 0, 1])), 27)
        self.assertEqual(env.obs2index(np.array([4, 4, 2])), 74)
        self.assertEqual(env.obs2index(np.array([2, 3, 2])), 67)
        self.assertEqual(env.obs2index(np.array([4, 4, 4])), 124)

        self.assertRaises(AssertionError, env.obs2index, np.array([4, 4, 5]))


class TestChainDomain(unittest.TestCase):
    """ tests the chain domain """

    def test_reset(self):
        """ tests ChainDomain.reset """
        domain = chain_domain.ChainDomain(size=4)

        obs = domain.reset()
        np.testing.assert_array_equal(domain.state, [0, 3])
        np.testing.assert_array_equal(obs, [0, 0, 0, 1, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0])

        domain = chain_domain.ChainDomain(size=10)

        obs = domain.reset()
        np.testing.assert_array_equal(domain.state, [0, 9])

        expected_obs = np.zeros((10, 10))
        expected_obs[0, 9] = 1
        np.testing.assert_array_equal(obs, expected_obs.reshape(100))

        domain.step(0)
        obs = domain.reset()
        np.testing.assert_array_equal(domain.state, [0, 9])
        np.testing.assert_array_equal(obs, expected_obs.reshape(100))

    def test_sample_start_state(self):
        """ tests sampling start states """

        env = chain_domain.ChainDomain(4)
        np.testing.assert_array_equal(env.sample_start_state(), [0, 3])

    def test_space(self):
        """ tests ChainDomain.spaces """

        domain = chain_domain.ChainDomain(size=3)

        action_space = domain.action_space

        self.assertEqual(action_space.n, 2)
        np.testing.assert_array_equal(action_space.size, [2])

        observation_space = domain.observation_space

        self.assertEqual(observation_space.n, 2 ** 9)
        np.testing.assert_array_equal(observation_space.size, [2] * 9)
        self.assertEqual(observation_space.ndim, 9)

    def test_state(self):
        """ tests ChainDomain.state """

        domain = chain_domain.ChainDomain(size=3)
        domain.reset()

        np.testing.assert_array_equal(domain.state, [0, 2])

        domain.step(0)
        self.assertIn(domain.state[0], [0, 1])
        self.assertEqual(domain.state[1], 1)

    def test_step(self):
        """ tests ChainDomain.step """

        domain = chain_domain.ChainDomain(size=3)

        # test if all effects go right
        step = domain.step(domain._action_mapping[0])  # pylint: disable=protected-access

        self.assertFalse(step.terminal)
        self.assertAlmostEqual(step.reward, -.0033333333)
        np.testing.assert_array_equal(
            step.observation, [0, 0, 0, 0, 1, 0, 0, 0, 0]
        )

        step = domain.step((domain._action_mapping[1]))  # pylint: disable=protected-access
        self.assertTrue(step.terminal)
        self.assertAlmostEqual(step.reward, 1 - .0033333333)
        np.testing.assert_array_equal(
            step.observation, [0, 0, 0, 0, 0, 0, 1, 0, 0]
        )

        domain.reset()

        # test going left
        step = domain.step(not domain._action_mapping[0])  # pylint: disable=protected-access
        self.assertFalse(step.terminal)
        self.assertAlmostEqual(step.reward, 0)
        np.testing.assert_array_equal(
            step.observation, [0, 1, 0, 0, 0, 0, 0, 0, 0]
        )

        step = domain.step(not domain._action_mapping[0])  # pylint: disable=protected-access
        self.assertTrue(step.terminal)
        self.assertAlmostEqual(step.reward, 0)
        np.testing.assert_array_equal(
            step.observation, [1, 0, 0, 0, 0, 0, 0, 0, 0]
        )

    def test_utils(self):
        """ tests ChainDomain utility functions

        ChainDomain.size
        ChainDomain.state2observation

        """

        for size in 3, 9:
            domain = chain_domain.ChainDomain(size=size)
            self.assertEqual(domain.size, size)

        domain = chain_domain.ChainDomain(size=4)

        expected_obs = np.zeros((4, 4))
        expected_obs[0, 0] = 1
        np.testing.assert_array_equal(
            domain.state2observation(np.array([0, 0])),
            expected_obs.reshape(16)
        )

        expected_obs = np.zeros((4, 4))
        expected_obs[2, 0] = 1
        np.testing.assert_array_equal(
            domain.state2observation(np.array([2, 0])),
            expected_obs.reshape(16)
        )

        expected_obs = np.zeros((4, 4))
        expected_obs[2, 3] = 1
        np.testing.assert_array_equal(
            domain.state2observation(np.array([2, 3])),
            expected_obs.reshape(16)
        )

    def test_observation_projection(self):
        """ tests obs2index """

        env = chain_domain.ChainDomain(size=4)

        def one_hot(obs):
            encoding = np.zeros((4, 4))
            encoding[obs[0], obs[1]] = 1

            return encoding.reshape(16)

        self.assertEqual(env.obs2index(one_hot(np.array([0, 0]))), 0)
        self.assertEqual(env.obs2index(one_hot(np.array([0, 1]))), 1)
        self.assertEqual(env.obs2index(one_hot(np.array([1, 0]))), 4)
        self.assertEqual(env.obs2index(one_hot(np.array([0, 2]))), 2)
        self.assertEqual(env.obs2index(one_hot(np.array([2, 3]))), 11)
        self.assertEqual(env.obs2index(one_hot(np.array([3, 2]))), 14)

        obs = np.zeros((5, 5))
        obs[4, 4] = 1
        self.assertRaises(AssertionError, env.obs2index, obs.reshape(25))


if __name__ == '__main__':
    unittest.main()
