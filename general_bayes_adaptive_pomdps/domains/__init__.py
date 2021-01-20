""" all the domains in which an agent can act """

import numpy as np
from general_bayes_adaptive_pomdps.environments import EncodeType, Environment

from .chain_domain import ChainDomain
from .collision_avoidance import CollisionAvoidance
from .gridverse_domain import GridverseDomain
from .gridworld import GridWorld
from .learned_environments import BADDr  # NOQA, ignore unused import
from .priors import (
    CollisionAvoidancePrior,
    GridWorldPrior,
    Prior,
    RoadRacerPrior,
    TigerPrior,
)
from .road_racer import RoadRacer
from .tiger import Tiger


def create_environment(
    domain_name: str,
    domain_size: int,
    encoding: EncodeType,
    description: str = "",
) -> Environment:
    """the factory function to construct environments

    Args:
         domain_name: (`str`): determines which domain is created
         domain_size: (`int`): the size of the domain (domain dependent)
         encoding: (`general_bayes_adaptive_pomdps.environments.EncodeType`): the encoding

    RETURNS (`general_bayes_adaptive_pomdps.environments.Environment`)

    """

    if domain_name == "tiger":
        return Tiger(encoding)
    if domain_name == "gridworld":
        return GridWorld(domain_size, encoding)
    if domain_name == "collision_avoidance":
        return CollisionAvoidance(domain_size)
    if domain_name == "chain":
        return ChainDomain(domain_size, encoding)
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
    encoding: EncodeType,
) -> Prior:
    """Builds the prior associated with input parameters

    Args:
         domain_name: (`str`): currently only accepting tiger
         domain_size: (`int`): size of domain
         prior_certainty: (`float`): Certainty of the prior: total number of counts
         prior_correctness: (`float`): Correctness of the prior: Tiger -> [0,1] -> [62.5, 85] observation probability
         encoding: (`EncodeType`): what observation encoding to use

    RETURNS (`general_bayes_adaptive_pomdps.domains.priors.Prior`):

    """

    if domain_name == "tiger":
        return TigerPrior(prior_certainty, prior_correctness, encoding)
    if domain_name == "gridworld":
        return GridWorldPrior(domain_size, encoding)
    if domain_name == "collision_avoidance":
        return CollisionAvoidancePrior(domain_size, prior_certainty)
    if domain_name == "road_racer":
        return RoadRacerPrior(domain_size, prior_certainty)

    raise ValueError("no known priors for domain " + domain_name)
