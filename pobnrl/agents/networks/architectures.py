""" various architectures for NNs """

import abc

from tensorflow.python.layers.layers import dense
from tensorflow.python.layers.layers import flatten
import tensorflow as tf


class Architecture(abc.ABC):
    """ implementation a q-function """

    @abc.abstractmethod
    def __call__(self, net_input, n_actions, scope):
        """ returns qvalues """

    @abc.abstractmethod
    def is_recurrent(self):
        """ returns whether it contains recurrent state """


class TwoHiddenLayerQNet(Architecture):
    """ Regular Q network with 2 hidden layers """

    _sizes = {'small': 16, 'med': 64, 'large': 512}

    # FIXME: take specific arguments instead of conf
    def __init__(self, conf):
        """ conf.network_size is in {'small', 'med', 'large'} """
        self.n_units = self._sizes[conf.network_size]

    def is_recurrent(self):
        return False

    def __call__(self, net_input, n_actions, scope):
        # concat all inputs but keep batch dimension
        hidden = flatten(net_input)

        # [fixme] programmed without really understanding what is happening
        # it should be possible to call this multiple times
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            print('Using network in scope', scope)
            hidden = dense(
                hidden,
                units=self.n_units,
                activation=tf.nn.tanh,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )
            hidden = dense(
                hidden,
                units=self.n_units,
                activation=tf.nn.tanh,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )

            qvalues = dense(
                hidden,
                units=n_actions,
                activation=None,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )

        return qvalues


class TwoHiddenLayerRecQNet(Architecture):
    """ Recurrent Q network with 2 hidden layers """

    _sizes = {'small': 16, 'med': 64, 'large': 512}
    rec_state = {}  # recurrent state for each scope

    def __init__(self, conf):
        """ conf.network_size is in {'small', 'med', 'large'} """
        self.n_units = self._sizes[conf.network_size]

    def is_recurrent(self):
        return True

    def __call__(self, net_input, n_actions, scope):
        # assume size of input is [batch size, history len, net_input...]
        assert len(net_input.shape) > 2

        batch_size = tf.shape(net_input)[0]
        history_len = tf.shape(net_input)[1]
        observation_num = net_input.shape[2:].num_elements()

        hidden = tf.reshape(  # flatten input but keep batch size and history len
            net_input,
            [batch_size, history_len, observation_num]
        )

        # [fixme] programmed without really understanding what is happening
        # it should be possible to call this multiple times
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            print('Using network in scope', scope)
            hidden = dense(
                hidden,
                units=self.n_units,
                activation=tf.nn.tanh,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )
            hidden = dense(
                hidden,
                units=self.n_units,
                activation=tf.nn.tanh,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )

            rnn_cell = tf.nn.rnn_cell.LSTMCell(
                self.n_units,
                initializer=tf.glorot_normal_initializer()
            )

            # can be initialized with a feed dict if you want to set this to a
            # previous state
            self.rec_state[scope] = rnn_cell.zero_state(batch_size, tf.float32)

            # will automatically handle the history len of each batch as a
            # single sequence
            hidden, new_rec_state = tf.nn.dynamic_rnn(
                rnn_cell,
                inputs=hidden,
                initial_state=self.rec_state[scope]
            )

            qvalues = dense(
                hidden[:, -1],
                units=n_actions,
                activation=None,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )

        return qvalues, new_rec_state
