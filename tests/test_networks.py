""" runs tests on the neural networks """

import unittest

import copy
import numpy as np
import torch

from po_nrl.agents.neural_networks.misc import perturb
from po_nrl.agents.neural_networks.networks import Net
from po_nrl.agents.neural_networks.neural_pomdps import DynamicsModel, sgd_builder, adam_builder, get_optimizer_builder
from po_nrl.environments import ActionSpace
from po_nrl.misc import DiscreteSpace


class TestDynamicModel(unittest.TestCase):
    """ Test unit for the `po_nrl.agents.neural_networks.neural_pomdps.DynamicsModel` """

    def is_equal_models(self, model_a, model_b, is_equal: bool) -> None:
        """ checks whether provided models are equal

        Args:
             model_a:
             model_b:
             is_equal: (`bool`)

        RETURNS (`None`):

        """

        test = self.assertTrue if is_equal else self.assertFalse

        for tensor_a, tensor_b in zip(
                model_a.parameters(),
                model_b.parameters()):
            test(torch.equal(tensor_a.data, tensor_b.data), f'{tensor_a} vs {tensor_b}')

    def test_freeze(self) -> None:
        """ tests whether freezing models works properly

        If freeze_O then the observation should not change with update (T should)
        If freeze_T then the observation should not change with update (O should)
        """

        test_model = DynamicsModel(
            state_space=DiscreteSpace([2]),
            action_space=ActionSpace(2),
            obs_space=DiscreteSpace([2]),
            network_size=5,
            learning_rate=.01,
            batch_size=0,
            dropout_rate=0.5,
            optimizer_builder=sgd_builder
        )

        copied_model = copy.deepcopy(test_model)
        test_model.perturb_parameters(freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_O)

        self.is_equal_models(test_model.net_t, copied_model.net_t, False)
        self.is_equal_models(test_model.net_o, copied_model.net_o, True)

        copied_model = copy.deepcopy(test_model)
        test_model.perturb_parameters(freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_T)

        self.is_equal_models(test_model.net_t, copied_model.net_t, True)
        self.is_equal_models(test_model.net_o, copied_model.net_o, False)

        copied_model = copy.deepcopy(test_model)
        test_model.batch_update(np.array([[0]]), np.array([0]), np.array([[0]]), np.array([[0]]), DynamicsModel.FreezeModelSetting.FREEZE_O)

        self.is_equal_models(test_model.net_t, copied_model.net_t, False)
        self.is_equal_models(test_model.net_o, copied_model.net_o, True)

        copied_model = copy.deepcopy(test_model)

        self.is_equal_models(test_model.net_t, copied_model.net_t, True)
        test_model.batch_update(np.array([[0]]), np.array([0]), np.array([[0]]), np.array([[0]]), DynamicsModel.FreezeModelSetting.FREEZE_T)

        self.is_equal_models(test_model.net_o, copied_model.net_o, False)
        self.is_equal_models(test_model.net_t, copied_model.net_t, True)

    def test_copy(self) -> None:
        """ tests the copy function of the `po_nrl.agents.neural_networks.neural_pomdps.DynamicsModel`

        Basically double checking whether the standard implementation works as
        **I** expect

        Args:

        RETURNS (`None`):

        """

        test_model = DynamicsModel(
            state_space=DiscreteSpace([2]),
            action_space=ActionSpace(2),
            obs_space=DiscreteSpace([2]),
            network_size=5,
            learning_rate=.01,
            batch_size=0,
            dropout_rate=0.5,
            optimizer_builder=sgd_builder
        )

        copied_model = copy.deepcopy(test_model)

        self.is_equal_models(test_model.net_t, copied_model.net_t, True)
        self.is_equal_models(test_model.net_o, copied_model.net_o, True)

        test_model.perturb_parameters()

        self.is_equal_models(test_model.net_t, copied_model.net_t, False)
        self.is_equal_models(test_model.net_o, copied_model.net_o, False)

        # TODO: set optimizer
        test_model = DynamicsModel(
            state_space=DiscreteSpace([2]),
            action_space=ActionSpace(1),
            obs_space=DiscreteSpace([2]),
            network_size=5,
            learning_rate=.01,
            batch_size=0,
            dropout_rate=0.5,
            optimizer_builder=sgd_builder
        )

        copied_model = copy.deepcopy(test_model)

        copied_model.batch_update(np.array([[1]]), np.array([0]), np.array([[0]]), np.array([[1]]))

        self.is_equal_models(test_model.net_t, copied_model.net_t, False)
        self.is_equal_models(test_model.net_o, copied_model.net_o, False)

    def test_optimizer_builder(self) -> None:
        """ Simple tests to ensure the correct builder is returned """
        self.assertEqual(get_optimizer_builder('SGD'), sgd_builder)
        self.assertEqual(get_optimizer_builder('Adam'), adam_builder)
        self.assertRaises(ValueError, get_optimizer_builder, 'a wrong value')

class TestMisc(unittest.TestCase):
    """ Tests `po_nrl.agents.neural_networks.misc` """

    def test_perturbations(self) -> None:  # pylint: disable=no-self-use
        """ tests `po_nrl.agents.neural_networks.misc.perturb`

        Args:

        RETURNS (`None`):

        """

        tensor = torch.rand((5, 3))

        perturbed_tensor = perturb(tensor, 1)

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            tensor.numpy(),
            perturbed_tensor.numpy()
        )


class TestNetwork(unittest.TestCase):
    """ tests some properties of the network """

    def test_dropout(self) -> None:
        """ some basic sanity checks of dropout functionality """

        net_input = torch.tensor([.1, 4., -.2])

        no_dropout = Net(input_size=3, output_size=2, layer_size=10, prior_scaling=0, dropout_rate=0)
        self.assertTrue(torch.eq(no_dropout(net_input), no_dropout(net_input)).all())

        no_dropout = Net(input_size=3, output_size=2, layer_size=10, prior_scaling=0, dropout_rate=.5)
        self.assertFalse(torch.eq(no_dropout(net_input), no_dropout(net_input)).all())


if __name__ == '__main__':
    unittest.main()
