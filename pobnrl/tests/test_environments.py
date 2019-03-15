""" runs tests on the environments """

import unittest

from environments import tiger


class TestTiger(unittest.TestCase):
    """ tests functionality of the tiger environment """

    @classmethod
    def setUpClass(cls):
        """ creates a tiger member """
        cls.env = tiger.Tiger(False)

    def test_start_state(self):
        """ tests that start state is 0 or 1 """
        state = self.env.sample_start_state()
        self.assertTrue(state in [0, 1])

    def test_step(self):
        """ tests some basic dynamics """
        self.assertFalse(True)

    def test_space(self):
        """ tests the size of the spaces """
        spaces = self.env.spaces()
        self.assertListEqual(spaces['A'].dimensions().tolist(), [3])
        self.assertListEqual(spaces['O'].dimensions().tolist(), [1, 1])


if __name__ == '__main__':
    unittest.main()
