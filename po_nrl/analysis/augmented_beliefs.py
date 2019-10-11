""" provides analyzers for augmented beliefs """

from typing import Union, Tuple, List
from functools import partial
import numpy as np

from po_nrl.agents.planning.particle_filters import ParticleFilter
from po_nrl.agents.planning.belief import BeliefAnalysis


def tiger_model_analysis(
        belief: ParticleFilter) -> List[Tuple[str, Union[float, np.ndarray]]]:
    """ inspects the belief over the observation model

    Collects the prediction of hearing correctly from 100 sampled models

    Args:
         belief: (`ParticleFilter`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """
    # hear correct for state 0
    hear_correct_0 = np.array([
        belief.sample().model.observation_model([0], 2, [0])[0][0]
        for _ in range(1000)
    ])

    # hear correct for state 1
    hear_correct_1 = np.array([
        belief.sample().model.observation_model([1], 2, [1])[0][1]
        for _ in range(1000)
    ])

    # tiger stays left when opening
    stay_left_prob = np.array([
        belief.sample().model.transition_model([0], 2)[0][0]
        for _ in range(1000)
    ])

    # tiger stays left when opening
    stay_right_prob = np.array([
        belief.sample().model.transition_model([1], 2)[0][1]
        for _ in range(1000)
    ])

    return [
        ('correct-observe-left', hear_correct_0),
        ('correct-observe-right', hear_correct_1),
        ('correct-transition-left', stay_left_prob),
        ('correct-transition-right', stay_right_prob),
    ]


def ca_transition_analysis(
        belief: ParticleFilter,
        size: int) -> List[Tuple[str, Union[float, np.ndarray]]]:
    """ provides analysis on belief over collision avoidance transition model

    returns the probability of correctly predicting the (deterministic) x pos

    Args:
         size: (`int`):
         belief: (`ParticleFilter`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """

    sample_size = 100
    states = np.stack((
        np.random.randint(1, size, sample_size),
        np.random.randint(0, size, size=sample_size),
        np.random.randint(0, size, size=sample_size)
    ), axis=1)

    actions = np.random.randint(0, 3, size=sample_size)

    correct_x_prob = np.array([
        belief.sample().model.transition_model(states[i], actions[i])[0][states[i][0] - 1]
        for i in range(sample_size)
    ], dtype=int)

    return [
        ('correct-x', correct_x_prob)
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


def analyzer_factory(domain: str, domain_size: int) -> BeliefAnalysis:
    """ factory for belief analysers

    Args:
         domain: (`str`):
         domain_size: (`int`):

    RETURNS(`BeliefAnalysis`):

    """

    if domain == 'tiger':
        return chain_analysis([tiger_model_analysis, count_unique_models])
    if domain == 'collision_avoidance':
        return chain_analysis([
            partial(ca_transition_analysis, size=domain_size),
            count_unique_models
        ])

    # default analysis
    return count_unique_models
