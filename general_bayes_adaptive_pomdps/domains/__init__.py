""" all the domains in which an agent can act """

import numpy as np

from general_bayes_adaptive_pomdps.core import Domain, DomainPrior

from .collision_avoidance import CollisionAvoidance, CollisionAvoidancePrior
from .gridverse_domain import GridverseDomain
from .gridworld import GridWorld, GridWorldPrior
from .road_racer import RoadRacer, RoadRacerPrior
from .tiger import Tiger, TigerPrior


def create_domain(
    domain_name: str,
    domain_size: int,
    use_one_hot_encoding: bool = False,
    description: str = "",
) -> Domain:
    """The factory function to construct domains

    `use_one_hot_encoding` depends on the chosen `domain_name`, but generally
    refers to using a one-hot encoding to represent part of either the state or
    observation. Examples:

        - :class:`Tiger`: observation (0/1/2 => 2 elements)
        - :class:`GridWorld`: goal representation

    Args:
         domain_name: (`str`): determines which domain is created
         domain_size: (`int`): the size of the domain (domain dependent)
         use_one_hot_encoding: (`bool`): whether to apply one-hot encoding where appropriate (domain dependent)

    RETURNS (`general_bayes_adaptive_pomdps.core.Domain`)

    """

    if domain_name == "tiger":
        return Tiger(use_one_hot_encoding)
    if domain_name == "gridworld":
        return GridWorld(domain_size, use_one_hot_encoding)
    if domain_name == "collision_avoidance":
        return CollisionAvoidance(domain_size)
    if domain_name == "road_racer":
        return RoadRacer(np.arange(1, domain_size + 1) / (domain_size + 1))
    if domain_name == "gridverse":

        if not description:
            return GridverseDomain()

        return GridverseDomain(*description.split(" "))

    raise ValueError("unknown domain " + domain_name)


def create_prior(
    domain_name: str,
    domain_size: int,
    prior_certainty: float,
    prior_correctness: float,
    use_one_hot_encoding: bool,
) -> DomainPrior:
    """Builds the prior associated with input parameters

    Args:
         domain_name: (`str`): currently only accepting tiger
         domain_size: (`int`): size of domain
         prior_certainty: (`float`): Certainty of the prior: total number of counts
         prior_correctness: (`float`): Correctness of the prior: Tiger -> [0,1] -> [62.5, 85] observation probability
         use_one_hot_encoding: (`bool`): whether to apply one-hot encoding where appropriate (domain dependent)

    RETURNS (`general_bayes_adaptive_pomdps.domains.priors.Prior`):

    """

    if domain_name == "tiger":
        return TigerPrior(prior_certainty, prior_correctness, use_one_hot_encoding)
    if domain_name == "gridworld":
        return GridWorldPrior(domain_size, use_one_hot_encoding)
    if domain_name == "collision_avoidance":
        return CollisionAvoidancePrior(domain_size, prior_certainty)
    if domain_name == "road_racer":
        return RoadRacerPrior(domain_size, prior_certainty)

    raise ValueError("no known priors for domain " + domain_name)
