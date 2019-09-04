""" All types of Neural nets used by agent as models """

import tensorflow as tf

from misc import Space
from environments import ActionSpace

from .networks import Net, simple_fc_rnn  # NOQA, ignore unused import
from .q_functions import QNetInterface, QNet, DRQNNet

from .misc import ReplayBuffer  # NOQA, ignore unused import


def create_qnet(
        action_space: ActionSpace,
        observation_space: Space,
        scope: str,
        conf) -> QNetInterface:
    """ factory for creating Q networks / policies

    Args:
         action_space: (`pobnrl.environments.ActionSpace`): of environment
         observation_space: (`pobnrl.misc.Space`): of environment
         scope: (`str`): name of agent (learning scope)
         conf: (`namespace`): configurations (see -h)

    RETURNS (`pobnrl.agents.neural_networks.q_functions.QNetInterface`):

    """

    if conf.recurrent:
        return DRQNNet(
            action_space,
            observation_space,
            simple_fc_rnn,
            tf.compat.v1.train.AdamOptimizer(conf.learning_rate),
            scope,
            conf,
        )

    return QNet(
        action_space,
        observation_space,
        conf,
    )
