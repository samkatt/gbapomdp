""" Contains networks """

from typing import Tuple, Optional
import torch


class Net(torch.nn.Module):  # type: ignore
    """ a standard 3-layer neural network """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_size: int,
            prior_scaling: float = 0,
            dropout_rate: float = 0):

        assert input_size > 0
        assert output_size > 0
        assert layer_size > 0
        assert prior_scaling >= 0
        assert 0 <= dropout_rate < 1

        super().__init__()

        self.prior_scaling = prior_scaling
        self.prior = torch.nn.Linear(input_size, output_size) if prior_scaling != 0 else None

        if self.prior:
            self.prior.requires_grad_(False)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.layer_1 = torch.nn.Linear(input_size, layer_size)
        self.layer_2 = torch.nn.Linear(layer_size, layer_size)

        self.out = torch.nn.Linear(layer_size, output_size)

        self.random_init_parameters()

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        """ forward passes through the network

        Args:
             net_input: (`torch.Tensor`):

        """
        activations = self.dropout(torch.tanh(self.layer_1(net_input)))
        activations = self.dropout(torch.tanh(self.layer_2(activations)))

        if self.prior:
            return self.out(activations) + self.prior_scaling * self.prior(net_input)

        return self.out(activations)

    def random_init_parameters(self) -> None:
        """ randomly resets / initiates parameters

        Args:

        RETURNS (`None`):

        """

        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()
        self.out.reset_parameters()

        if self.prior:
            self.prior.reset_parameters()


class RecNet(torch.nn.Module):  # type: ignore
    """ a standard 3-layer recurrent neural network """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_size: int,
            prior_scaling: float = 0):

        super().__init__()

        self.prior_scaling = prior_scaling
        self.prior = torch.nn.LSTM(input_size, output_size, batch_first=True) if prior_scaling != 0 else None

        if self.prior:
            self.prior.requires_grad_(False)

        self.layer_1 = torch.nn.Linear(input_size, layer_size)
        self.layer_2 = torch.nn.Linear(layer_size, layer_size)

        self.rnn_layer = torch.nn.LSTM(layer_size, layer_size, batch_first=True)
        self.out = torch.nn.Linear(layer_size, output_size)

        self.random_init_parameters()

    def forward(
            self,
            net_input: torch.Tensor,
            hidden: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """ forward passes through the network

        Will pass the network input with optional hidden state and return the output plus new hidden state

        Args:
             net_input: (`torch`):
             Tensor:
             hidden: (`Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]]`): hidden state of `this`

        RETURNS (`Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor,torch.Tensor],Optional[Tuple[torch.Tensor,torch.Tensor]]]]`):

        """

        hidden_net: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        hidden_prior: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if hidden:
            hidden_net, hidden_prior = hidden

        activations = torch.tanh(self.layer_1(net_input))
        activations = torch.tanh(self.layer_2(activations))

        # XXX: use packing for increased performance
        activations, hidden_net = self.rnn_layer(activations, hidden_net)

        assert hidden_net

        if self.prior:

            # XXX: use packing for increased performance
            prior, hidden_prior = self.prior(net_input, hidden_prior)

            assert hidden_prior

            return (
                self.out(activations) + self.prior_scaling * prior,
                (hidden_net, hidden_prior)
            )

        # no prior
        assert not hidden_prior

        return (
            self.out(activations),
            (hidden_net, None)
        )

    def random_init_parameters(self) -> None:
        """ randomly resets / initiates parameters

        Args:

        RETURNS (`None`):

        """

        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()
        self.out.reset_parameters()
        self.rnn_layer.reset_parameters()

        if self.prior:
            self.prior.reset_parameters()
