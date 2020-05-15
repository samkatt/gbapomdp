""" all the domains in which an agent can act """

import numpy as np

from po_nrl.environments import Environment, EncodeType

from .priors import Prior

from .cartpole import Cartpole
from .chain_domain import ChainDomain
from .collision_avoidance import CollisionAvoidance
from .gridworld import GridWorld
from .learned_environments import NeuralEnsemblePOMDP  # NOQA, ignore unused import
from .road_racer import RoadRacer
from .tiger import Tiger

from .priors import TigerPrior, GridWorldPrior, CollisionAvoidancePrior, RoadRacerPrior


def create_environment(
        domain_name: str,
        domain_size: int,
        encoding: EncodeType) -> Environment:
    """ the factory function to construct environmentss

    Args:
         domain_name: (`str`): determines which domain is created
         domain_size: (`int`): the size of the domain (domain dependent)
         encoding: (`po_nrl.environments.EncodeType`): the encoding

    RETURNS (`po_nrl.environments.Environment`)

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
    if domain_name == "road_racer":
        return RoadRacer(np.arange(1, domain_size + 1) / (domain_size + 1))

    raise ValueError('unknown domain ' + domain_name)


def create_prior(
        domain_name: str,
        domain_size: int,
        prior_certainty: float,
        prior_correctness: float,
        encoding: EncodeType) -> Prior:
    """ Builds the prior associated with input parameters

    Args:
         domain_name: (`str`): currently only accepting tiger
         domain_size: (`int`): size of domain
         prior_certainty: (`float`): Certainty of the prior: Tiger, CollisionAvoidance, RoadRacer -> total number of counts
         prior_correctness: (`float`): Correctness of the prior: Tiger -> [0,1] -> [62.5, 85] observation probability
         encoding: (`EncodeType`): what observation encoding to use

    RETURNS (`po_nrl.domains.priors.Prior`):

    """

    if domain_name == "tiger":
        return TigerPrior(prior_certainty, prior_correctness, encoding)
    if domain_name == "gridworld":
        return GridWorldPrior(domain_size, encoding)
    if domain_name == 'collision_avoidance':
        return CollisionAvoidancePrior(domain_size, prior_certainty)
    if domain_name == 'road_racer':
        return RoadRacerPrior(domain_size, prior_certainty)

    raise ValueError('no known priors for domain ' + domain_name)
