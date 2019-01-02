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

class TwoHiddenLayerQNet(Architecture):
    """ Regular Q network with 2 hidden layers """

    _sizes = {'small': 16, 'med': 64, 'large': 512}

    def __init__(self, conf):
        """ conf.network_size is in {'small', 'med', 'large'} """
        self.n_units = self._sizes[conf.network_size]

    def __call__(self, net_input, n_actions, scope):
        hidden = flatten(net_input)

        with tf.variable_scope(scope):
            hidden = dense(hidden, units=self.n_units, activation=tf.nn.tanh) # first layer
            hidden = dense(hidden, units=self.n_units, activation=tf.nn.tanh) # second layer
            qvalues = dense(hidden, units=n_actions, activation=None)

        return qvalues
