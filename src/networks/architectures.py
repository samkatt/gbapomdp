""" various architectures for NNs """

import abc
import tensorflow as tf
from tensorflow.python.layers.layers import dense
from tensorflow.python.layers.layers import flatten

class QNet(abc.ABC):
    """ implementation a q-function """

    @abc.abstractmethod
    def __call__(self, net_input, n_actions, scope):
        """ returns qvalues """
        pass

class TwoHiddenLayerQNet(QNet):
    """ Regular Q network with 2 hidden layers """

    def __call__(self, net_input, n_actions, scope):
        hidden = flatten(net_input)

        with tf.variable_scope(scope):
            hidden = dense(hidden, units=512, activation=tf.nn.tanh) # first layer
            hidden = dense(hidden, units=512, activation=tf.nn.tanh) # second layer
            qvalues = dense(hidden, units=n_actions, activation=None)

        return qvalues
