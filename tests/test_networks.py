""" runs tests on the neural networks """

import unittest

import copy
import numpy as np
import torch

from po_nrl.agents.neural_networks.misc import perturb
from po_nrl.agents.neural_networks.networks import Net
from po_nrl.agents.neural_networks.neural_pomdps import DynamicsModel
from po_nrl.environments import ActionSpace
from po_nrl.misc import DiscreteSpace


class TestDynamicModel(unittest.TestCase):
    """ Test unit fo the `po_nrl.agents.neural_networks.neural_pomdps.DynamicsModel` """

    def test_copy(self) -> None:
        """ tests the copy function of the `po_nrl.agents.neural_networks.neural_pomdps.DynamicsModel`

        Basically double checking whether the standard implementation works as
        **I** expect

        Args:

        RETURNS (`None`):

        """

        test_model = DynamicsModel(
            state_space=DiscreteSpace([1]),
            action_space=ActionSpace(1),
            obs_space=DiscreteSpace([1]),
            network_size=5,
            learning_rate=.01,
            batch_size=0,
            dropout_rate=0.5
        )

        copied_model = copy.deepcopy(test_model)

        for test_tensor, copy_tensor in zip(
                test_model.net_t.parameters(),
                copied_model.net_t.parameters()):
            self.assertTrue(torch.equal(test_tensor.data, copy_tensor.data))

        test_model.perturb_parameters()

        for test_tensor, copy_tensor in zip(
                test_model.net_t.parameters(),
                copied_model.net_t.parameters()):
            self.assertFalse(torch.equal(test_tensor.data, copy_tensor.data))

        test_model = DynamicsModel(
            state_space=DiscreteSpace([2]),
            action_space=ActionSpace(1),
            obs_space=DiscreteSpace([2]),
            network_size=5,
            learning_rate=.01,
            batch_size=0,
            dropout_rate=0.5
        )
        copied_model = copy.deepcopy(test_model)

        copied_model.batch_update(np.array([[0]]), np.array([0]), np.array([[0]]), np.array([[0]]))

        for test_tensor, copy_tensor in zip(
                test_model.net_t.parameters(),
                copied_model.net_t.parameters()):
            self.assertFalse(torch.equal(test_tensor.data, copy_tensor.data))


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
