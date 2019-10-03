""" All types of Neural nets used by agent as models """

from po_nrl.misc import Space
from po_nrl.environments import ActionSpace

from .networks import Net, RecNet  # NOQA, ignore unused import
from .q_functions import QNetInterface, QNet, RecQNet

from .misc import ReplayBuffer  # NOQA, ignore unused import


def create_qnet(
        action_space: ActionSpace,
        observation_space: Space,
        conf,
        name: str) -> QNetInterface:
    """ factory for creating Q networks / policies

    Args:
         action_space: (`pobnrl.environments.ActionSpace`): of environment
         observation_space: (`pobnrl.misc.Space`): of environment
         scope: (`str`): name of agent (learning scope)
         conf: (`namespace`): configurations (see -h)
         name: (`str`): name of net

    RETURNS (`pobnrl.agents.neural_networks.q_functions.QNetInterface`):

    """

    if conf.recurrent:
        return RecQNet(
            action_space,
            observation_space,
            conf,
            name
        )

    return QNet(
        action_space,
        observation_space,
        conf,
        name
    )
