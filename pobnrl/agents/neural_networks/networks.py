""" Contains networks """

from tensorflow.keras.layers import Flatten, Dense
import tensorflow as tf

import torch
import torch.nn
import torch.functional


def simple_fc_nn(net_input, n_out: int, n_hidden: int):
    """ construct a fully connected nn of a two-hidden layer architecture

    Assumes size of input is [batch size, history len, net_input...]

    Args:
         net_input: the input of the network (observation)
         n_out: (`int`): # of actions
         n_hidden: (`int`): # of units per layer

    """

    hidden = Flatten()(net_input)  # concat all inputs but keep batch dimension

    # it should be possible to call this multiple times
    with tf.compat.v1.variable_scope(""):

        with tf.compat.v1.variable_scope('layer'):
            for layer in range(2):  # 2 hidden layers
                hidden = Dense(units=n_hidden, activation='tanh', name=str(layer))(hidden)

        out = Dense(units=n_out, name='out')(hidden)

    return out


def simple_fc_rnn(
        net_input,
        init_rnn_state,
        n_out: int,
        n_hidden: int):
    """ constructs a fully connected rnn of a two-hidden layer architecture

    scope must be unique to this network to ensure this works fine
    (tensorflow).

    Assumes size of input is [batch size, history len, net_input...]

    Args:
         net_input: the input of the network (observation)
         seq_lengths: the length of each batch
         rnn_cell: the actual rnn component (cell) of the rec layer
         init_rnn_state: state of the recurrent layer
         n_out: (`int`): # of actions
         n_hidden: (`int`): # of units per layer

    """

    assert len(net_input.shape) > 2

    batch_size = tf.shape(net_input)[0]
    history_len = tf.shape(net_input)[1]
    observation_num = net_input.shape[2:].num_elements()

    # flatten but keep batch size and history len
    hidden = tf.reshape(
        net_input,
        [batch_size, history_len, observation_num]
    )

    # it should be possible to call this multiple times
    with tf.compat.v1.variable_scope(""):

        with tf.compat.v1.variable_scope('layer'):
            for layer in range(2):  # 2 hidden layers
                hidden = Dense(units=n_hidden, activation='tanh', name=str(layer))(hidden)

        hidden = tf.keras.layers.Masking(mask_value=0., name='mask')(hidden)
        hidden, cell_state_h, cell_state_c = tf.keras.layers.LSTM(
            units=n_hidden,
            return_state=True,
            name="rnn"
        )(hidden, initial_state=init_rnn_state)

        out = Dense(units=n_out, name="out")(hidden)

    return out, [cell_state_h, cell_state_c]


class Net(torch.nn.Module):  # type: ignore
    """ a standard 3-layer neural network """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layer_size: int,
            prior_scaling: float = 0):

        super(Net, self).__init__()

        self.prior_scaling = prior_scaling
        self.prior = torch.nn.Linear(input_size, output_size)
        self.prior.requires_grad_(False)

        self.layer_1 = torch.nn.Linear(input_size, layer_size)
        self.layer_2 = torch.nn.Linear(layer_size, layer_size)

        self.out = torch.nn.Linear(layer_size, output_size)

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        """ forward passes through the network

        Args:
             net_input: (`torch.Tensor`):

        """
        activations = torch.functional.F.tanh(self.layer_1(net_input))
        activations = torch.functional.F.tanh(self.layer_2(activations))
        return self.out(activations) + self.prior_scaling * self.prior(net_input)

    def random_init_parameters(self) -> None:
        """ randomly resets / initiates parameters

        Args:

        RETURNS (`None`):

        """

        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()
        self.out.reset_parameters()
        self.prior.reset_parameters()
