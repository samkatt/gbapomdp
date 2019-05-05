""" all the domains in which an agent can act """

# from environments import Environment  # TODO: add after debugging works again

from .cartpole import Cartpole
from .chain_domain import ChainDomain
from .collision_avoidance import CollisionAvoidance
from .gridworld import GridWorld
from .tiger import Tiger


def create_environment(
        domain_name: str,
        domain_size: int,
        # verbose: int) -> Environment:  # TODO: add after debugging works again
        verbose: int):
    """ the factory function to constructenvironmentss

    TODO: rename to create_domains

    Args:
         domain_name: (`str`): determines which domain is created
         domain_size: (`int`): the size of the domain (domain dependent)
         verbose: (`int`): verbosity level

    RETURNS (`pobnrl.environments.Environment`)

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
