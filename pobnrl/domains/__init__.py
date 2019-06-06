""" all the domains in which an agent can act """

from enum import Enum

from environments import Environment

from .cartpole import Cartpole
from .chain_domain import ChainDomain
from .collision_avoidance import CollisionAvoidance
from .gridworld import GridWorld
from .tiger import Tiger


class EncodeType(Enum):
    """ AgentType """
    DEFAULT = 0
    ONE_HOT = 1


def create_environment(
        domain_name: str,
        domain_size: int,
        encoding: EncodeType) -> Environment:
    """ the factory function to construct environmentss

    Args:
         domain_name: (`str`): determines which domain is created
         domain_size: (`int`): the size of the domain (domain dependent)
         encoding: (`EncodeType`): the encoding

    RETURNS (`pobnrl.environments.Environment`)

    """

    if domain_name == "tiger":
        return Tiger(use_one_hot=encoding == EncodeType.ONE_HOT)
    if domain_name == "cartpole":
        return Cartpole()
    if domain_name == "gridworld":
        return GridWorld(domain_size, one_hot_goal_encoding=encoding == EncodeType.ONE_HOT)
    if domain_name == "collision_avoidance":
        return CollisionAvoidance(domain_size)
    if domain_name == "chain":
        return ChainDomain(domain_size)

    raise ValueError('unknown domain ' + domain_name)
