""" All types of Neural nets used by agent as models """

import tensorflow as tf

from misc import DiscreteSpace

from .neural_network_misc import two_layer_q_net, two_layer_rec_q_net
from .q_functions import QNetInterface, DQNNet, DRQNNet


def create_qnet(
        action_space: DiscreteSpace,
        observation_space: DiscreteSpace,
        scope: str,
        conf) -> QNetInterface:
    """ factory for creating Q networks / policies

    Args:
         action_space: (`pobnrl.misc.DiscreteSpace`): of environment
         observation_space: (`pobnrl.misc.DiscreteSpace`): of environment
         scope: (`str`): name of agent (learning scope)
         conf: (`namespace`): configurations (see -h)

    RETURNS (`pobnrl.agents.networks.q_functions.QNetInterface`):

    """

    if conf.recurrent:
        return DRQNNet(
            action_space,
            observation_space,
            two_layer_rec_q_net,
            tf.train.AdamOptimizer(conf.learning_rate),
            scope,
            conf,
        )

    return DQNNet(
        action_space,
        observation_space,
        two_layer_q_net,
        tf.train.AdamOptimizer(conf.learning_rate),
        scope,
        conf,
    )
