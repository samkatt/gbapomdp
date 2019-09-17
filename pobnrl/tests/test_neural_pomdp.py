""" runs tests on the neural pomdp package """

import unittest

from environments import ActionSpace
from misc import DiscreteSpace
from agents.neural_networks.neural_pomdps import DynamicsModel


class TestDynamicModel(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
