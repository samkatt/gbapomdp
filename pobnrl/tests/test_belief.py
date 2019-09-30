""" tests functionality of the agents.planning.belief module """

import unittest

from agents.planning import belief
from domains import tiger
from environments import EncodeType


class TestFactory(unittest.TestCase):
    """ tests the factory """

    def test_belief_param(self) -> None:
        """ tests basic stuff on the first (belief) parameter """

        # belief should be something legit
        self.assertRaises(AssertionError, belief.belief_update_factory, 'test', 0, 0, 0)

        # test returns rejection or importance sampling correctly
        self.assertEqual(belief.belief_update_factory('importance_sampling', 0, False, tiger.Tiger(EncodeType.DEFAULT)), belief.importance_sampling)

        self.assertEqual(
            belief.belief_update_factory(  # type: ignore
                'rejection_sampling', 0, False, tiger.Tiger(EncodeType.DEFAULT)
            ).func,
            belief.rejection_sampling
        )

        # backprop and rs
        bp_rs = belief.belief_update_factory(
            'rejection_sampling', 0, True, tiger.Tiger(EncodeType.DEFAULT)
        )

        self.assertEqual(
            bp_rs.func,  # type: ignore
            belief.augmented_rejection_sampling
        )

        self.assertEqual(
            bp_rs.keywords['update_model'],  # type: ignore
            belief.backprop_update
        )

        # perturb and importance
        bp_rs = belief.belief_update_factory(
            'importance_sampling', 0.1, False, tiger.Tiger(EncodeType.DEFAULT)
        )

        self.assertEqual(
            bp_rs.func,  # type: ignore
            belief.augmented_importance_sampling
        )

        self.assertEqual(
            bp_rs.keywords['update_model'].func,  # type: ignore
            belief.perturb_parameters
        )
