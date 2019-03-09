""" various architectures for NNs """

import abc

from tensorflow.python.layers.layers import dense
from tensorflow.python.layers.layers import flatten
import tensorflow as tf


class Architecture(abc.ABC):
    """ implementation a q-function """

    @abc.abstractmethod
    def __call__(self, net_input, n_actions: int, scope: str):
        """ computes the q values given the net input

        Returns n_actions q values

        Args:
             net_input: the input to the network
             n_actions: (`int`): the number of actions (outputs)
             scope: (`str`): the scope of the network (used by tensorflow)

        """

    @abc.abstractmethod
    def is_recurrent(self) -> bool:
        """ used to check whether the network is recurrent

        Can be useful in determining e.g. whether there is some internal state

        RETURNS (`bool`): whether the network is recurrent

        """


class TwoHiddenLayerQNet(Architecture):
    """ Regular Q network with 2 hidden layers """

    _sizes = {'small': 16, 'med': 64, 'large': 512}

    def __init__(self, network_size: str):
        """ construct the TwoHiddenLayerQNet of the specified size

        Translates 'small' to 16, 'med' to 64 and 'large' to 512 hidden notes

        Args:
             network_size: (`str`): is in {'small', 'med', 'large'}

        """
        self.n_units = self._sizes[network_size]

    def is_recurrent(self) -> bool:
        """ returns false since this network is not recurrent

        RETURNS (`bool`): false (TwoHiddenLayerQNet is not recurrent)

        """
        return False

    def __call__(self, net_input, n_actions: int, scope: str):
        """ returns n_actions Q-values given the network input

        scope must be unique to this network to ensure this works fine (tensorflow). This is the
        main functionality of any network

        Assumes size of input is [batch size, history len, net_input...]

        Args:
             net_input: (tensor) input to the network
             n_actions: (`int`): number of outputs (actions)
             scope: (`str`): the (unique) scope of the network used by tensorflow

        """
        # concat all inputs but keep batch dimension
        hidden = flatten(net_input)

        # FIXME: programmed without really understanding what is happening
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

    def __init__(self, network_size: int):
        """ construct this of the specified size

        Translates 'small' to 16, 'med' to 64 and 'large' to 512 hidden notes

        Args:
             network_size: (`str`): is in {'small', 'med', 'large'}

        """
        self.n_units = self._sizes[network_size]

    def is_recurrent(self) -> bool:
        """ returns true

        interface implementation of Architecture

        RETURNS (`bool`): true, as this is recurrent

        """
        return True

    def __call__(self, net_input, n_actions: int, scope: str):
        """ returns n_actions Q-values given the network input

        This is the main functionality of any network
        scope must be unique to this network to ensure this works fine (tensorflow).

        Assumes size of input is [batch size, history len, net_input...]

        Args:
             net_input: (tensor) input to the network
             n_actions: (`int`): number of outputs (actions)
             scope: (`str`): the (unique) scope of the network used by tensorflow

        """
        assert len(net_input.shape) > 2

        batch_size = tf.shape(net_input)[0]
        history_len = tf.shape(net_input)[1]
        observation_num = net_input.shape[2:].num_elements()

        hidden = tf.reshape(  # flatten input but keep batch size and history len
            net_input,
            [batch_size, history_len, observation_num]
        )

        # FIXME: programmed without really understanding what is happening
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
