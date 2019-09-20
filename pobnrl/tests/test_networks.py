""" runs tests on the neural networks """

import unittest

import torch
import numpy as np

from environments import ActionSpace
from misc import DiscreteSpace
from agents.neural_networks.neural_pomdps import DynamicsModel
from agents.neural_networks.misc import perturb


class TestDynamicModel(unittest.TestCase):
    """ Test unit fo the `pobnrl.agents.neural_networks.neural_pomdps.DynamicsModel` """

    def test_copy(self) -> None:
        """ tests the copy function of the `pobnrl.agents.neural_networks.neural_pomdps.DynamicsModel`

        Args:

        RETURNS (`None`):

        """

        test_model = DynamicsModel(
            state_space=DiscreteSpace([1]),
            action_space=ActionSpace(1),
            obs_space=DiscreteSpace([1]),
            network_size=5,
            learning_rate=.01,
            name='test'
        )

        copied_model = test_model.copy()
        self.assertEqual(test_model.name, 'test')
        self.assertNotEqual(test_model.net_t, copied_model.net_t)

        self.assertEqual(copied_model.name, 'test-copy-1')

        nested_copied_model = copied_model.copy()
        self.assertEqual(nested_copied_model.name, 'test-copy-2')

        # test when there is a number present
        model_with_numbered_name = DynamicsModel(
            state_space=DiscreteSpace([1]),
            action_space=ActionSpace(1),
            obs_space=DiscreteSpace([1]),
            network_size=5,
            learning_rate=.01,
            name='some-model-3'
        )

        copied_model = model_with_numbered_name.copy()
        self.assertEqual(copied_model.name, 'some-model-3-copy-1')

        nested_copied_model = copied_model.copy()
        self.assertEqual(nested_copied_model.name, 'some-model-3-copy-2')


class TestMisc(unittest.TestCase):
    """ Tests `pobnrl.agents.neural_networks.misc` """

    def test_perturbations(self) -> None:  # pylint: disable=no-self-use
        """ tests `pobnrl.agents.neural_networks.misc.perturb`

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


if __name__ == '__main__':
    unittest.main()
