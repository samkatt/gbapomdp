""" Contains networks """

import torch


class Net(torch.nn.Module):
    """a standard 3-layer neural network"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_size: int,
        prior_scaling: float = 0,
        dropout_rate: float = 0,
    ):

        assert input_size > 0
        assert output_size > 0
        assert layer_size > 0
        assert prior_scaling >= 0
        assert 0 <= dropout_rate < 1

        super().__init__()

        self.prior_scaling = prior_scaling
        self.prior = (
            torch.nn.Linear(input_size, output_size) if prior_scaling != 0 else None
        )

        if self.prior:
            self.prior.requires_grad_(False)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.layer_1 = torch.nn.Linear(input_size, layer_size)
        self.layer_2 = torch.nn.Linear(layer_size, layer_size)
        self.layer_3 = torch.nn.Linear(layer_size, layer_size)

        self.out = torch.nn.Linear(layer_size, output_size)

        self.random_init_parameters()

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        """forward passes through the network

        Args:
             net_input: (`torch.Tensor`):

        """
        activations = self.dropout(torch.tanh(self.layer_1(net_input)))
        activations = torch.tanh(self.layer_2(activations))
        activations = torch.tanh(self.layer_3(activations))

        if self.prior:
            return self.out(activations) + self.prior_scaling * self.prior(net_input)

        return self.out(activations)

    def random_init_parameters(self) -> None:
        """randomly resets / initiates parameters

        Args:

        RETURNS (`None`):

        """

        for layer in [self.layer_1, self.layer_2, self.out]:
            layer.reset_parameters()

        if self.prior:
            self.prior.reset_parameters()
