""" runs tests on particle_filters"""

import unittest
import random

from po_nrl.agents.planning.particle_filters import FlatFilter, WeightedFilter, WeightedParticle, resample


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


class TestResampling(unittest.TestCase):
    """ tests `po_nrl.agents.planning.particle_filters.resample` """

    def test_num_particles(self) -> None:
        """ tests it returns the correct number of samples """

        wfilter = WeightedFilter()

        wfilter.add_particle(500)
        wfilter.add_particle(4)
        wfilter.add_particle(2)
        wfilter = resample(wfilter)

        self.assertEqual(wfilter.size, 3)

        wfilter.add_particle(100)
        wfilter = resample(wfilter)

        self.assertEqual(wfilter.size, 4)

    def test_total_weight(self) -> None:
        """ tests the total weight of resampled filter == N """

        wfilter = WeightedFilter()

        size = random.randint(5, 10)
        for i in range(size):
            wfilter.add_weighted_particle(WeightedParticle(i, random.random()))
        wfilter = resample(wfilter)

        self.assertEqual(wfilter._total_weight, size)  # pylint: disable=protected-access

    def test_basic_probability(self) -> None:
        """ tests some basic functionality

        * a single particle filter will return itself
        * filter with 1 highly likely particle will return filter with that particle
        """

        wfilter = WeightedFilter()

        wfilter.add_weighted_particle(WeightedParticle(500, 100))
        particles = list(resample(wfilter).particles)

        self.assertEqual(len(particles), 1)
        self.assertEqual(particles[0].weight, 1)
        self.assertEqual(particles[0].value, 500)

        wfilter.add_weighted_particle(WeightedParticle(10, .00001))
        particles = list(resample(wfilter).particles)

        self.assertEqual(len(particles), 2)
        self.assertEqual(particles[0].weight, 1)
        self.assertEqual(particles[0].value, 500)
        self.assertEqual(particles[1].weight, 1)
        self.assertEqual(particles[1].value, 500)


if __name__ == '__main__':
    unittest.main()
