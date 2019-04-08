""" all the environments in which an agent can act """

from .environment import Environment
from .tiger import Tiger
from .cartpole import Cartpole
from .collision_avoidance import CollisionAvoidance
from .gridworld import GridWorld
from .chain_domain import ChainDomain


def create_environment(
        domain_name: str,
        domain_size: int,
        verbose: int) -> Environment:
    """ the factory function to construct environments

    Args:
         domain_name: (`str`): determines which domain is created
         domain_size: (`int`): the size of the domain (domain dependent)
         verbose: (`int`): verbosity level

    RETURNS (`pobnrl.environments.environment.Environment`)

    """
    verbose = verbose > 1

    if domain_name == "tiger":
        return Tiger(verbose)
    if domain_name == "cartpole":
        return Cartpole(verbose)
    if domain_name == "gridworld":
        return GridWorld(domain_size, verbose)
    if domain_name == "collision_avoidance":
        return CollisionAvoidance(domain_size, verbose)
    if domain_name == "chain":
        return ChainDomain(domain_size, verbose)

    raise ValueError('unknown domain ' + domain_name)
