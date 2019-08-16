""" all the domains in which an agent can act """

from environments import Environment, EncodeType

from .priors import Prior  # NOQA, ignore unused import

from .cartpole import Cartpole
from .chain_domain import ChainDomain
from .collision_avoidance import CollisionAvoidance
from .gridworld import GridWorld
from .learned_environments import NeuralEnsemblePOMDP  # NOQA, ignore unused import
from .tiger import Tiger

from .priors import TigerPrior, GridWorldPrior, CollisionAvoidancePrior


def create_environment(
        domain_name: str,
        domain_size: int,
        encoding: EncodeType) -> Environment:
    """ the factory function to construct environmentss

    Args:
         domain_name: (`str`): determines which domain is created
         domain_size: (`int`): the size of the domain (domain dependent)
         encoding: (`pobnrl.environments.EncodeType`): the encoding

    RETURNS (`pobnrl.environments.Environment`)

    """

    if domain_name == "tiger":
        return Tiger(encoding)
    if domain_name == "cartpole":
        return Cartpole()
    if domain_name == "gridworld":
        return GridWorld(domain_size, encoding)
    if domain_name == "collision_avoidance":
        return CollisionAvoidance(domain_size)
    if domain_name == "chain":
        return ChainDomain(domain_size, encoding)

    raise ValueError('unknown domain ' + domain_name)


def create_prior(
        domain_name: str,
        domain_size: int,
        encoding: EncodeType) -> Prior:
    """ create_prior

    Args:
         domain_name: (`str`): currently only accepting tiger
         _domain_size: (`int`): unused atm
         encoding: (`EncodeType`): what observation encoding to use

    RETURNS (`pobnrl.domains.priors.Prior`):

    """

    if domain_name == "tiger":
        return TigerPrior(encoding)
    if domain_name == "gridworld":
        return GridWorldPrior(domain_size, encoding)
    if domain_name == 'collision_avoidance':
        return CollisionAvoidancePrior(domain_size)

    raise ValueError('no known priors for domain ' + domain_name)
