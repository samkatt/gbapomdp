""" runs tests on particle_filters"""

import unittest

from agents.planning.particle_filters import FlatFilter, WeightedFilter, WeightedParticle
from agents.planning.particle_filters import rejection_sampling, importance_sampling
from agents.planning.particle_filters import resample


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

        samples = [ffilter.sample() for _ in range(15)]
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

        wfilter.add_particle(5)
        self.assertEqual(wfilter.sample(), 5)

        wfilter.add_weighted_particle(WeightedParticle(5, 1))
        self.assertEqual(wfilter.sample(), 5)

        wfilter.add_weighted_particle(WeightedParticle(1, 10000000))
        self.assertEqual(wfilter.sample(), 1)


class TestBeliefUpdates(unittest.TestCase):
    """ tests belief updates """

    @staticmethod
    def increment(some_num: int):
        """ increments input int `some_num` """
        return some_num + 1

    @staticmethod
    def accept_all(_):
        """ returns true for all input """
        return True

    @staticmethod
    def one(_):
        """ returns 1 for all input """
        return 1

    @staticmethod
    def over_one(some_num: int):
        """ returns 1/`sum_num` """
        return 1 / float(some_num)

    @staticmethod
    def is_odd(some_num: int):
        """ returns true if input int x is odd """
        return not some_num % 2 == 0

    def test_rejection_sampling(self):
        """ tests the rejection sampling way of updating belief """

        ffilter = FlatFilter()
        for _ in range(5):
            ffilter.add_particle(0)

        ffilter1 = rejection_sampling(ffilter, self.increment, self.accept_all)

        self.assertEqual(ffilter1.sample(), 1)

        ffilter1.add_particle(10)
        ffilter2 = rejection_sampling(ffilter1, self.increment, self.is_odd)

        self.assertEqual(ffilter2.sample(), 11)
        self.assertIn(ffilter1.sample(), [1, 10])

    def test_importance_sampling(self):
        """ tests the importance sampling way of updating belief """

        wfilter = WeightedFilter()
        for _ in range(5):
            wfilter.add_weighted_particle(WeightedParticle(value=1, weight=.5))

        wfilter1 = importance_sampling(wfilter, self.increment, self.one)

        for particle in wfilter1:
            self.assertEqual(particle.weight, .5)
            self.assertEqual(particle.value, 2)

        wfilter1.add_weighted_particle(WeightedParticle(value=10, weight=5))
        wfilter2 = importance_sampling(wfilter1, self.increment, self.over_one)

        for particle in wfilter2:
            self.assertIn(particle.value, [11, 3])

            if particle.value == 11:
                self.assertEqual(particle.weight, 5 * (1 / 11))
            else:
                self.assertAlmostEqual(particle.weight, .5 * (1 / 3))


class TestResampling(unittest.TestCase):
    """ tests particle_filters.resample """

    def test_basic(self) -> None:
        """ some basic easy test with 1 particle """

        # 1 particle flat filter
        flat_filter = FlatFilter()
        flat_filter.add_particle(5)
        new_filter = resample(flat_filter, 3)

        self.assertIsInstance(new_filter, FlatFilter)
        self.assertEqual(new_filter.size, 3)
        for part in new_filter:
            self.assertEqual(part, 5)

        # 1 particle weighted filter
        weighted_filter = WeightedFilter()
        weighted_filter.add_weighted_particle(WeightedParticle(3, .2))
        new_filter = resample(weighted_filter, 2)

        self.assertIsInstance(new_filter, WeightedFilter)
        self.assertEqual(new_filter.size, 2)
        for part in new_filter:
            self.assertEqual(part.value, 3)

        # 2 particles of which one with 0 weight weighted filter
        weighted_filter.add_weighted_particle(WeightedParticle(100, .0))
        new_filter = resample(weighted_filter, 10)

        self.assertIsInstance(new_filter, WeightedFilter)
        self.assertEqual(new_filter.size, 10)
        for part in new_filter:
            self.assertEqual(part.value, 3)


if __name__ == '__main__':
    unittest.main()
