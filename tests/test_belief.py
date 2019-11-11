""" tests functionality of the agents.planning.belief module """

from functools import partial
import unittest

from po_nrl.agents.neural_networks.neural_pomdps import DynamicsModel
from po_nrl.agents.planning import belief
from po_nrl.domains import tiger
from po_nrl.environments import EncodeType
from po_nrl.model_based import parse_arguments

# pylint: disable=no-member


class TestFactory(unittest.TestCase):
    """ tests the factory """

    @staticmethod
    def create_conf(
            belief_type: str,
            perturb_stdev: float,
            backprop: bool,
            replay_update: bool,
            freeze_model_setting: str,
            sample_size: int):
        """ generates a complete configuration file given the input settings

        Args:
             belief_type: (`str`):
             perturb_stdev: (`float`):
             backprop: (`bool`):
             replay_update: (`bool`):
             freeze_model_setting: (`str`):
             sample_size: (`int`)

        """

        config_list = [
            '-D=tiger',   # dummy domain
            f'-B={belief_type}',
            f'--perturb_stdev={perturb_stdev}',
            f'--belief_minimal_sample_size={sample_size}'
        ]

        if backprop:
            config_list.append('--backprop')
        if replay_update:
            config_list.append('--replay_update')
        if freeze_model_setting:
            config_list.append(f'--freeze_model={freeze_model_setting}')

        conf = parse_arguments(config_list)

        return conf

    def test_belief_param(self) -> None:
        """ tests basic stuff on the first (belief) parameter """

        # # test returns rejection or importance sampling correctly
        self.assertEqual(
            belief.belief_update_factory(  # type: ignore
                TestFactory.create_conf('importance_sampling', 0, False, False, "", 20),
                tiger.Tiger(EncodeType.DEFAULT)).func,
            belief.importance_sampling
        )

        # # test returns rejection or importance sampling correctly
        self.assertEqual(
            belief.belief_update_factory(  # type: ignore
                TestFactory.create_conf('importance_sampling', 0, False, False, "", 20),
                tiger.Tiger(EncodeType.DEFAULT)).keywords['minimal_sampling_size'],
            20
        )

        self.assertEqual(
            belief.belief_update_factory(  # type: ignore
                TestFactory.create_conf('rejection_sampling', 0, False, False, "", 20), tiger.Tiger(EncodeType.DEFAULT)
            ).func,
            belief.rejection_sampling
        )

        # backprop and rs
        bp_rs = belief.belief_update_factory(
            TestFactory.create_conf('rejection_sampling', 0, True, False, "", 20), tiger.Tiger(EncodeType.DEFAULT)
        )

        self.assertEqual(
            bp_rs.func,  # type: ignore
            belief.augmented_rejection_sampling
        )

        self.assertEqual(len(bp_rs.keywords['update_model'].model_updates), 1)  # type: ignore

        self.assertEqual(
            bp_rs.keywords['update_model'].model_updates[0].args,  # type: ignore
            partial(belief.backprop_update, freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_NONE).args
        )
        self.assertEqual(
            bp_rs.keywords['update_model'].model_updates[0].func,  # type: ignore
            partial(belief.backprop_update, freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_NONE).func
        )

        # perturb and importance
        bp_is = belief.belief_update_factory(
            TestFactory.create_conf('importance_sampling', 0.1, False, False, "", 10), tiger.Tiger(EncodeType.DEFAULT)
        )

        self.assertEqual(
            bp_is.func,  # type: ignore
            belief.augmented_importance_sampling
        )
        self.assertEqual(bp_is.keywords['minimal_sampling_size'], 30)  # type: ignore
        self.assertEqual(len(bp_is.keywords['update_model'].model_updates), 1)  # type: ignore
        self.assertEqual(
            bp_is.keywords['update_model'].model_updates[0].func,  # type: ignore
            partial(
                belief.perturb_parameters,
                stdev=.1,
                freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_NONE
            ).func
        )
        self.assertEqual(
            bp_is.keywords['update_model'].model_updates[0].args,  # type: ignore
            partial(
                belief.perturb_parameters,
                stdev=.1,
                freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_NONE
            ).args
        )
        self.assertEqual(
            bp_is.keywords['update_model'].model_updates[0].keywords,  # type: ignore
            partial(
                belief.perturb_parameters,
                stdev=.1,
                freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_NONE
            ).keywords
        )

        # perturb and backprop
        is_rs_bp = belief.belief_update_factory(
            TestFactory.create_conf('importance_sampling', .1, True, False, "T", 100), tiger.Tiger(EncodeType.DEFAULT)
        )

        self.assertEqual(len(is_rs_bp.keywords['update_model'].model_updates), 2)  # type: ignore
        self.assertEqual(
            is_rs_bp.keywords['update_model'].model_updates[0].func,  # type: ignore
            partial(belief.backprop_update, freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_T).func
        )
        self.assertEqual(
            is_rs_bp.keywords['update_model'].model_updates[0].args,  # type: ignore
            partial(belief.backprop_update, freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_T).args
        )
        self.assertEqual(
            is_rs_bp.keywords['update_model'].model_updates[1].func,  # type: ignore
            partial(belief.perturb_parameters, stdev=.1, freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_T).func
        )
        self.assertEqual(
            is_rs_bp.keywords['update_model'].model_updates[1].args,  # type: ignore
            partial(belief.perturb_parameters, stdev=.1, freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_T).args
        )
