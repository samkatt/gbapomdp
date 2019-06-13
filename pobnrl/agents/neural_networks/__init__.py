""" All types of Neural nets used by agent as models """

import tensorflow as tf

from misc import DiscreteSpace
from environments import ActionSpace

from .networks import simple_fc_nn, simple_fc_rnn
from .q_functions import QNetInterface, DQNNet, DRQNNet

from .misc import ReplayBuffer  # NOQA, ignore unused import


def create_qnet(
        action_space: ActionSpace,
        observation_space: DiscreteSpace,
        scope: str,
        conf) -> QNetInterface:
    """ factory for creating Q networks / policies

    Args:
         action_space: (`pobnrl.environments.ActionSpace`): of environment
         observation_space: (`pobnrl.misc.DiscreteSpace`): of environment
         scope: (`str`): name of agent (learning scope)
         conf: (`namespace`): configurations (see -h)

    RETURNS (`pobnrl.agents.neural_networks.q_functions.QNetInterface`):

    """

    if conf.recurrent:
        return DRQNNet(
            action_space,
            observation_space,
            simple_fc_rnn,
            tf.train.AdamOptimizer(conf.learning_rate),
            scope,
            conf,
        )

    return DQNNet(
        action_space,
        observation_space,
        simple_fc_nn,
        tf.train.AdamOptimizer(conf.learning_rate),
        scope,
        conf,
    )
