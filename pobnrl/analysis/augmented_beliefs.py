""" provides analyzers for augmented beliefs """

from typing import Union, Tuple, List
import numpy as np

from agents.planning.particle_filters import ParticleFilter
from agents.planning.belief import BeliefAnalysis, noop_analysis
from misc import POBNRLogger


def tiger_observation_model_analyse(
        belief: ParticleFilter) -> List[Tuple[str, Union[float, np.ndarray]]]:
    """ inspects the belief over the observation model

    Collects the prediction of hearing correctly from 100 sampled models

    Args:
         belief: (`ParticleFilter`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """
    # hear correct for state 0
    probs_0 = np.array([
        belief.sample().model.observation_prob([0], 2, [0], [0])
        for _ in range(100)
    ])

    # hear correct for state 1
    probs_1 = np.array([
        belief.sample().model.observation_prob([1], 2, [1], [1])
        for _ in range(100)
    ])

    return [
        ('correct-observe-left', probs_0),
        ('correct-observe-right', probs_1),
    ]


def count_unique_models(
        belief: ParticleFilter) -> List[Tuple[str, Union[float, np.ndarray]]]:
    """ returns number of unique models from 100 samples

    Args:
         belief: (`ParticleFilter`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """

    num_unique = len({belief.sample().model for _ in range(100)})

    return [('unique_models_per_100', num_unique)]


def chain_analysis(to_chain: List[BeliefAnalysis]) -> BeliefAnalysis:
    """ chains `BeliefAnalysis` into a single call

    Args:
         to_chain: (`List[BeliefAnalysis]`):

    RETURNS (`BeliefAnalysis`):

    """

    def chain(belief: ParticleFilter) -> List[Tuple[str, Union[float, np.ndarray]]]:

        chained_analysis: List[Tuple[str, Union[float, np.ndarray]]] = []

        for analysis in to_chain:
            chained_analysis.extend(analysis(belief))

        return chained_analysis

    return chain


def analyzer_factory(domain: str) -> BeliefAnalysis:
    """ factory for belief analysers

    Args:
         domain: (`str`):

    RETURNS(`BeliefAnalysis`):

    """

    if domain == 'tiger':
        return chain_analysis([tiger_observation_model_analyse, count_unique_models])

    # default analysis
    return count_unique_models
