"""tests :mod:`general_bayes_adaptive_pomdps.baddr.neural_networks.misc`"""

import numpy as np
import pytest
import torch

from general_bayes_adaptive_pomdps.baddr.neural_networks.misc import (
    ReplayBuffer,
    perturb,
    whiten_input,
)


class TestReplayBuffer:
    """ Tests some functionality of the replay buffer """

    def test_capacity(self):
        """ tests the capacity property of the replay buffer """

        replay_buffer = ReplayBuffer()

        assert replay_buffer.capacity == 5000

        replay_buffer.store((), False)
        assert replay_buffer.capacity == 5000

        replay_buffer.store((), False)
        assert replay_buffer.capacity == 5000

        replay_buffer.store((), True)
        assert replay_buffer.capacity == 5000

    def test_size(self):
        """ tests the size property of replay buffer """

        replay_buffer = ReplayBuffer()

        assert replay_buffer.size == 1

        replay_buffer.store((), False)
        assert replay_buffer.size == 1

        replay_buffer.store((), False)
        assert replay_buffer.size == 1

        replay_buffer.store((), True)
        assert replay_buffer.size == 2

        replay_buffer.store((), False)
        assert replay_buffer.size == 2

        replay_buffer.store((), True)
        replay_buffer.store((), True)
        assert replay_buffer.size == 4

        replay_buffer.store((), True)
        assert replay_buffer.size == 5


class TestMisc:
    """ Tests `general_bayes_adaptive_pomdps.baddr.neural_networks.misc` """

    def test_whiten_input(self) -> None:
        assert whiten_input(np.array([0]), np.random.random()) == -1
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

        assert whiten_input(2, 8) == -0.5
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


if __name__ == "__main__":
    pytest.main([__file__])
