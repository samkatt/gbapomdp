""" tests functionality of the agents.planning.belief module """

from functools import partial
import unittest

from po_nrl.agents.planning import belief
from po_nrl.domains import tiger
from po_nrl.environments import EncodeType


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
        self.assertListEqual(
            bp_rs.keywords['update_model'].model_updates,  # type: ignore
            [belief.backprop_update]
        )

        # perturb and importance
        bp_rs = belief.belief_update_factory(
            'importance_sampling', 0.1, False, tiger.Tiger(EncodeType.DEFAULT)
        )

        self.assertEqual(
            bp_rs.func,  # type: ignore
            belief.augmented_importance_sampling
        )
        self.assertEqual(len(bp_rs.keywords['update_model'].model_updates), 1)  # type: ignore
        self.assertEqual(
            bp_rs.keywords['update_model'].model_updates[0].func,  # type: ignore
            partial(belief.perturb_parameters, stdev=.1).func
        )
        self.assertEqual(
            bp_rs.keywords['update_model'].model_updates[0].args,  # type: ignore
            partial(belief.perturb_parameters, stdev=.1).args
        )
        self.assertEqual(
            bp_rs.keywords['update_model'].model_updates[0].keywords,  # type: ignore
            partial(belief.perturb_parameters, stdev=.1).keywords
        )

        # perturb and backprop
        is_rs_bp = belief.belief_update_factory(
            'importance_sampling', .1, True, tiger.Tiger(EncodeType.DEFAULT)
        )

        self.assertEqual(len(is_rs_bp.keywords['update_model'].model_updates), 2)  # type: ignore
        self.assertEqual(
            is_rs_bp.keywords['update_model'].model_updates[0],  # type: ignore
            belief.backprop_update
        )
        self.assertEqual(
            is_rs_bp.keywords['update_model'].model_updates[1].func,  # type: ignore
            partial(belief.perturb_parameters, stdev=.1).func
        )
