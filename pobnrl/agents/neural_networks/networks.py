""" Contains networks """

from tensorflow.keras.layers import Flatten, Dense
import tensorflow as tf


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

        print(out)

    return out, [cell_state_h, cell_state_c]
