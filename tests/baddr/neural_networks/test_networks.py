"""tests :mod:`general_bayes_adaptive_pomdps.baddr.neural_networks.networks`"""

import pytest
import torch

from general_bayes_adaptive_pomdps.baddr.neural_networks.networks import Net


class TestNetwork:
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
        assert torch.eq(no_dropout(net_input), no_dropout(net_input)).all()

        no_dropout = Net(
            input_size=3,
            output_size=2,
            layer_size=10,
            prior_scaling=0,
            dropout_rate=0.5,
        )
        assert not torch.eq(no_dropout(net_input), no_dropout(net_input)).all()


if __name__ == "__main__":
    pytest.main([__file__])
