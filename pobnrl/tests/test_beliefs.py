""" runs tests on beliefs"""

import unittest

from agents.planning import FlatFilter, WeightedFilter
from agents.planning.beliefs import WeightedParticle


class TestFilters(unittest.TestCase):
    """ class to test filters """

    def test_flat_filter(self):
        """ tests sampling from a flat filter """

        ffilter = FlatFilter()

        self.assertRaises(IndexError, ffilter.sample)

        ffilter.add_particle(5)
        self.assertEqual(ffilter.sample(), 5)

        ffilter.add_particle(5)
        self.assertEqual(ffilter.sample(), 5)

        ffilter.add_particle(2)

        samples = [ffilter.sample() for _ in range(10)]
        self.assertIn(2, samples)
        self.assertIn(5, samples)

        for sample in samples:
            self.assertIn(sample, [2, 5])

    def test_weighted_particle(self):
        """ tests the weighted particle """

        self.assertRaises(ValueError, WeightedParticle, 2, -.1)

        particle = WeightedParticle(5, 1.6)
        self.assertEqual(particle.weight, 1.6)
        self.assertEqual(particle.value, 5)

    def test_weighted_filter(self):
        """ tests sampling from a weighted filter """

        wfilter = WeightedFilter()
        self.assertRaises(IndexError, wfilter.sample)

        wfilter.add_particle(WeightedParticle(5, 1))
        self.assertEqual(wfilter.sample(), 5)

        wfilter.add_particle(WeightedParticle(5, 1))
        self.assertEqual(wfilter.sample(), 5)

        wfilter.add_particle(WeightedParticle(1, 10000000))
        self.assertEqual(wfilter.sample(), 1)


if __name__ == '__main__':
    unittest.main()
