""" Contains networks """

from tensorflow.python.layers.layers import dense
from tensorflow.python.layers.layers import flatten
import tensorflow as tf


def simple_fc_nn(net_input, n_out: int, n_hidden: int):
    """ construct a fully connected nn of a two-hidden layer architecture

    Assumes size of input is [batch size, history len, net_input...]

    Args:
         net_input: the input of the network (observation)
         n_out: (`int`): # of actions
         n_hidden: (`int`): # of units per layer

    """

    hidden = flatten(net_input)  # concat all inputs but keep batch dimension

    # it should be possible to call this multiple times
    with tf.compat.v1.variable_scope(
            tf.compat.v1.get_default_graph().get_name_scope(),
            reuse=tf.compat.v1.AUTO_REUSE):

        with tf.compat.v1.variable_scope('hidden'):
            for layer in range(2):  # 2 hidden layers
                hidden = dense(
                    hidden,
                    units=n_hidden,
                    activation=tf.nn.tanh,
                    name=str(layer)
                )

        out = dense(
            hidden,
            units=n_out,
            name="out"
        )

    return out


def simple_fc_rnn(
        net_input,
        seq_lengths,
        rnn_cell,
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
    with tf.compat.v1.variable_scope(
            tf.compat.v1.get_default_graph().get_name_scope(),
            reuse=tf.compat.v1.AUTO_REUSE):

        with tf.compat.v1.variable_scope('hidden'):
            for layer in range(2):  # 2 hidden layers
                hidden = dense(
                    hidden,
                    units=n_hidden,
                    activation=tf.nn.tanh,
                    name=str(layer)
                )

        # handlse the history len of each batch as a single sequence
        hidden, new_rec_state = tf.nn.dynamic_rnn(
            rnn_cell,
            sequence_length=seq_lengths,
            inputs=hidden,
            initial_state=init_rnn_state,
            dtype=tf.float32,
            scope="rnn"
        )

        seq_q_mask = tf.stack(
            [tf.range(tf.size(seq_lengths)), seq_lengths - 1],
            axis=-1
        )

        out = dense(
            tf.gather_nd(hidden, seq_q_mask),
            units=n_out,
            name="out"
        )

    return out, new_rec_state
