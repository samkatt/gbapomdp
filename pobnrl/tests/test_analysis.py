""" tests basic functionality of the analysis module """

import unittest

import numpy as np

from analysis.merge_result_files import extract_statistics, combine_var_and_mean


class TestMerging(unittest.TestCase):
    """ tests the merging component of the analysis module """

    def test_extracting_statistics(self) -> None:
        """ tests what happens if you merge same contents """

        file_a = np.array([0, 1, 2, 3])
        file_b = np.array([4, 5, 6, 7])

        extracted = extract_statistics([file_a, file_b])

        self.assertEqual(extracted[0]['mu'], 0)
        self.assertEqual(extracted[0]['var'], 1)
        self.assertEqual(extracted[0]['n'], 2)

        self.assertEqual(extracted[1]['mu'], 4)
        self.assertEqual(extracted[1]['var'], 5)
        self.assertEqual(extracted[1]['n'], 6)

    def test_combining_same_file(self) -> None:
        """ tests simple case of merging the same file """

        stats_a = {'mu': 1., 'var': 2., 'n': 3}

        amount_of_files = np.random.randint(2, 10)
        combined = combine_var_and_mean([stats_a] * amount_of_files)

        self.assertEqual(combined['mu'], 1.)
        self.assertLess(combined['var'], 2.)
        self.assertEqual(combined['n'], 3 * amount_of_files)

    def test_combining_stats(self) -> None:
        """ some defualt tests for combining stats """

        stats_a = {'mu': 1., 'var': 5., 'n': 1}
        stats_b = {'mu': 2., 'var': 3., 'n': 2}
        stats_c = {'mu': 10., 'var': 4., 'n': 6}

        combined = combine_var_and_mean([stats_a, stats_b, stats_c])

        self.assertAlmostEqual(combined['mu'], 7.222222222222222)
        self.assertEqual(combined['n'], 9)

    def test_combine_variance(self) -> None:
        """ taken from `https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html` """

        group_a = {'mu': 63., 'var': 81., 'n': 50.}
        group_b = {'mu': 54., 'var': 36., 'n': 40.}

        combined = combine_var_and_mean([group_a, group_b])

        self.assertEqual(combined['mu'], 59)
        self.assertEqual(combined['n'], 90)
        self.assertAlmostEqual(combined['var'], 80.59550561797735)
