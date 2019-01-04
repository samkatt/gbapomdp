""" various architectures for NNs """

import abc
import tensorflow as tf
from tensorflow.python.layers.layers import dense
from tensorflow.python.layers.layers import flatten

class Architecture(abc.ABC):
    """ implementation a q-function """

    @abc.abstractmethod
    def __call__(self, net_input, n_actions, scope):
        """ returns qvalues """
        pass

    @abc.abstractmethod
    def is_recursive(self):
        """ returns whether it contains recurrent state """
        pass

class TwoHiddenLayerQNet(Architecture):
    """ Regular Q network with 2 hidden layers """

    _sizes = {'small': 16, 'med': 64, 'large': 512}

    def __init__(self, conf):
        """ conf.network_size is in {'small', 'med', 'large'} """
        self.n_units = self._sizes[conf.network_size]

    def is_recursive(self):
        return False

    def __call__(self, net_input, n_actions, scope):
        hidden = flatten(net_input) # concat all inputs but keep batch dimension

        with tf.variable_scope(scope):
            hidden = dense(hidden, units=self.n_units, activation=tf.nn.tanh)
            hidden = dense(hidden, units=self.n_units, activation=tf.nn.tanh)
            qvalues = dense(hidden, units=n_actions, activation=None)

        return qvalues


class TwoHiddenLayerRecQNet(Architecture):
    """ Regular Q network with 2 hidden layers """

    _sizes = {'small': 16, 'med': 64, 'large': 512}

    def __init__(self, conf):
        """ conf.network_size is in {'small', 'med', 'large'} """
        self.n_units = self._sizes[conf.network_size]

    def is_recursive(self):
        return True

    def __call__(self, net_input, n_actions, scope):
        # assume size of input is [batch size, history len, net_input...]
        assert len(net_input.shape) > 2

        batch_size = tf.shape(net_input)[0]
        history_len = tf.shape(net_input)[1]
        observation_num = net_input.shape[2:].num_elements()

        hidden = tf.reshape( # flatten input but keep batch size and history len
            net_input,
            [batch_size, history_len, observation_num]
        )

        with tf.variable_scope(scope):
            hidden = dense(hidden, units=self.n_units, activation=tf.nn.tanh)
            hidden = dense(hidden, units=self.n_units, activation=tf.nn.tanh)

            rnn_cell = tf.nn.rnn_cell.LSTMCell(self.n_units)
            self.rec_state = rnn_cell.zero_state(batch_size, tf.float32)

            # will automatically handle the history len of each batch as a single sequence
            hidden, new_rec_state = tf.nn.dynamic_rnn(
                rnn_cell,
                inputs=hidden,
                initial_state=self.rec_state
            )

            qvalues = dense(hidden[:, -1], units=n_actions, activation=None)

        return qvalues, new_rec_state
