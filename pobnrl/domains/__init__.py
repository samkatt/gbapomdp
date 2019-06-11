""" all the domains in which an agent can act """

from enum import Enum

from environments import Environment

from .priors import Prior  # NOQA, ignore unused import

from .cartpole import Cartpole
from .chain_domain import ChainDomain
from .collision_avoidance import CollisionAvoidance
from .gridworld import GridWorld
from .tiger import Tiger

from .priors import TigerPrior


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

    # FIXME: probably cascade the type down into the domains
    if domain_name == "tiger":
        return Tiger(encoding == EncodeType.ONE_HOT)
    if domain_name == "cartpole":
        return Cartpole()
    if domain_name == "gridworld":
        return GridWorld(domain_size, encoding == EncodeType.ONE_HOT)
    if domain_name == "collision_avoidance":
        return CollisionAvoidance(domain_size)
    if domain_name == "chain":
        return ChainDomain(domain_size, encoding == EncodeType.ONE_HOT)

    raise ValueError('unknown domain ' + domain_name)


def create_prior(
        domain_name: str,
        _domain_size: int,
        encoding: EncodeType) -> Prior:
    """ create_prior

    Args:
         domain_name: (`str`): currently only accepting tiger
         _domain_size: (`int`): unused atm
         encoding: (`EncodeType`): what observation encoding to use

    RETURNS (`pobnrl.domains.priors.Prior`):

    """

    assert domain_name == "tiger", f'currently only suppert tiger, not {domain_name}'

    return TigerPrior(encoding == EncodeType.ONE_HOT)
