""" runs tests on the environments """

import unittest

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
        self.assertListEqual(spaces['A'].dimensions().tolist(), [3])
        self.assertListEqual(spaces['O'].dimensions().tolist(), [1, 1])


if __name__ == '__main__':
    unittest.main()
