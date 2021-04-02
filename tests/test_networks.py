""" runs tests on the neural networks """

import copy
import unittest

import numpy as np
import torch

from general_bayes_adaptive_pomdps.baddr.neural_networks.misc import (
    perturb,
    whiten_input,
)
from general_bayes_adaptive_pomdps.baddr.neural_networks.networks import Net
from general_bayes_adaptive_pomdps.baddr.neural_networks.neural_pomdps import (
    DynamicsModel,
    adam_builder,
    get_optimizer_builder,
    sgd_builder,
)
from general_bayes_adaptive_pomdps.core import ActionSpace
from general_bayes_adaptive_pomdps.misc import DiscreteSpace


class TestDynamicModel(unittest.TestCase):
    """ Test unit for the `general_bayes_adaptive_pomdps.baddr.neural_networks.neural_pomdps.DynamicsModel` """

    def setUp(self):
        s_space = DiscreteSpace([2])
        a_space = ActionSpace(2)
        o_space = DiscreteSpace([2])

        t_net = DynamicsModel.TNet(
            s_space,
            a_space,
            sgd_builder,
            learning_rate=0.1,
            network_size=5,
            dropout_rate=0.5,
        )
        o_net = DynamicsModel.ONet(
            s_space,
            a_space,
            o_space,
            sgd_builder,
            learning_rate=0.1,
            network_size=5,
            dropout_rate=0.5,
        )

        self.test_model = DynamicsModel(
            state_space=s_space,
            action_space=a_space,
            batch_size=0,
            t_model=t_net,
            o_model=o_net,
        )

    def is_equal_models(self, model_a, model_b, is_equal: bool) -> None:
        """checks whether provided models are equal

        Args:
             model_a:
             model_b:
             is_equal: (`bool`)

        RETURNS (`None`):

        """

        test = self.assertTrue if is_equal else self.assertFalse

        for tensor_a, tensor_b in zip(model_a.parameters(), model_b.parameters()):
            test(
                torch.equal(tensor_a.data, tensor_b.data),
                f"{tensor_a} vs {tensor_b}",
            )

    def test_freeze(self) -> None:
        """tests whether freezing models works properly

        If freeze_O then the observation should not change with update (T should)
        If freeze_T then the observation should not change with update (O should)
        """

        copied_model = copy.deepcopy(self.test_model)
        self.test_model.perturb_parameters(
            freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_O
        )

        self.is_equal_models(self.test_model.t.net, copied_model.t.net, False)  # type: ignore
        self.is_equal_models(self.test_model.o.net, copied_model.o.net, True)  # type: ignore

        copied_model = copy.deepcopy(self.test_model)
        self.test_model.perturb_parameters(
            freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_T
        )

        self.is_equal_models(self.test_model.t.net, copied_model.t.net, True)  # type: ignore
        self.is_equal_models(self.test_model.o.net, copied_model.o.net, False)  # type: ignore

        copied_model = copy.deepcopy(self.test_model)
        self.test_model.batch_update(
            np.array([[0.5]]),
            np.array([0]),
            np.array([[0]]),
            np.array([[0]]),
            conf=DynamicsModel.FreezeModelSetting.FREEZE_O,
        )

        self.is_equal_models(self.test_model.t.net, copied_model.t.net, False)  # type: ignore
        self.is_equal_models(self.test_model.o.net, copied_model.o.net, True)  # type: ignore

        copied_model = copy.deepcopy(self.test_model)

        self.is_equal_models(self.test_model.t.net, copied_model.t.net, True)  # type: ignore
        self.test_model.batch_update(
            np.array([[0]]),
            np.array([0]),
            np.array([[0]]),
            np.array([[0]]),
            conf=DynamicsModel.FreezeModelSetting.FREEZE_T,
        )

        self.is_equal_models(self.test_model.o.net, copied_model.o.net, False)  # type: ignore
        self.is_equal_models(self.test_model.t.net, copied_model.t.net, True)  # type: ignore

    def test_copy(self) -> None:
        """tests the copy function of the `general_bayes_adaptive_pomdps.baddr.neural_networks.neural_pomdps.DynamicsModel`

        Basically double checking whether the standard implementation works as
        **I** expect

        Args:

        RETURNS (`None`):

        """

        copied_model = copy.deepcopy(self.test_model)

        self.is_equal_models(
            self.test_model.t.net, copied_model.t.net, True  # type:ignore
        )
        self.is_equal_models(
            self.test_model.o.net, copied_model.o.net, True  # type:ignore
        )

        self.test_model.perturb_parameters()

        self.is_equal_models(
            self.test_model.t.net, copied_model.t.net, False  # type: ignore
        )
        self.is_equal_models(self.test_model.o.net, copied_model.o.net, False)  # type: ignore

        copied_model = copy.deepcopy(self.test_model)

        self.is_equal_models(self.test_model.t.net, copied_model.t.net, True)  # type: ignore
        self.is_equal_models(self.test_model.o.net, copied_model.o.net, True)  # type: ignore

        copied_model.batch_update(
            np.array([[1]]), np.array([0]), np.array([[0]]), np.array([[1]])
        )

        self.is_equal_models(self.test_model.t.net, copied_model.t.net, False)  # type: ignore
        self.is_equal_models(self.test_model.o.net, copied_model.o.net, False)  # type: ignore

    def test_optimizer_builder(self) -> None:
        """ Simple tests to ensure the correct builder is returned """
        self.assertEqual(get_optimizer_builder("SGD"), sgd_builder)
        self.assertEqual(get_optimizer_builder("Adam"), adam_builder)
        self.assertRaises(ValueError, get_optimizer_builder, "a wrong value")


class TestMisc(unittest.TestCase):
    """ Tests `general_bayes_adaptive_pomdps.baddr.neural_networks.misc` """

    def test_whiten_input(self) -> None:
        self.assertEqual(whiten_input(np.array([0]), np.random.random()), -1)
        np.testing.assert_array_equal(
            whiten_input(np.array([0, 5.5, 11]), 11), np.array([-1, 0, 1])
        )

        np.testing.assert_array_equal(
            whiten_input(np.array([1, 2, 3, 4], dtype=int), np.array([1, 2, 3, 4])),
            np.array([1, 1, 1, 1]),
        )
        np.testing.assert_array_equal(
            whiten_input(np.array([0, 0, 0, 0]), np.array([1, 2, 3, 4])),
            np.array([-1, -1, -1, -1]),
        )

        torch.equal(
            whiten_input(torch.tensor([0.5, 1, 1.5, 2]), torch.tensor([1, 2, 3, 4])),
            torch.tensor([0.0, 0.0, 0.0, 0.0]),
        )

        whitened_data = whiten_input(torch.tensor([0.1, 0.9, 1.1, 1.9]), 2)
        assert torch.less_equal(whitened_data, 1.0).all()
        assert torch.greater_equal(whitened_data, -1.0).all()
        assert torch.less_equal(whitened_data[:2], 0).all()
        assert torch.greater_equal(whitened_data[2:], 0).all()

        self.assertEqual(whiten_input(2, 8), -0.5)
        np.testing.assert_array_equal(
            whiten_input(np.array([2, 6], dtype=int), 8),
            np.array([-0.5, 0.5]),
        )

    def test_perturbations(self) -> None:
        """tests `general_bayes_adaptive_pomdps.baddr.neural_networks.misc.perturb`

        Args:

        RETURNS (`None`):

        """

        tensor = torch.rand((5, 3))

        perturbed_tensor = perturb(tensor, 1)

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            tensor.numpy(),
            perturbed_tensor.numpy(),
        )


class TestNetwork(unittest.TestCase):
    """ tests some properties of the network """

    def test_dropout(self) -> None:
        """ some basic sanity checks of dropout functionality """

        net_input = torch.tensor([0.1, 4.0, -0.2])

        no_dropout = Net(
            input_size=3,
            output_size=2,
            layer_size=10,
            prior_scaling=0,
            dropout_rate=0,
        )
        self.assertTrue(torch.eq(no_dropout(net_input), no_dropout(net_input)).all())

        no_dropout = Net(
            input_size=3,
            output_size=2,
            layer_size=10,
            prior_scaling=0,
            dropout_rate=0.5,
        )
        self.assertFalse(torch.eq(no_dropout(net_input), no_dropout(net_input)).all())


if __name__ == "__main__":
    unittest.main()
