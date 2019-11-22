""" provides analyzers for augmented beliefs """

from typing import Union, Tuple, List
from functools import partial
import numpy as np

from po_nrl.agents.planning.particle_filters import ParticleFilter
from po_nrl.agents.planning.belief import BeliefAnalysis

from po_nrl.domains.road_racer import RoadRacer


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

    returns the probability of
        - (deterministic) x position
        - (deterministic) y position
        - impossible obstacle transition
        - obstacle staying in position (mid or bottom)

    Args:
         size: (`int`):
         belief: (`ParticleFilter`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """

    mid = int(size / 2)
    sample_size = 100
    actions = np.random.randint(0, 3, size=sample_size)

    non_terminal_states = np.stack((
        np.random.randint(1, size, sample_size),
        np.random.randint(0, size, size=sample_size),
        np.random.randint(0, size, size=sample_size)
    ), axis=1)

    correct_next_y = np.clip(non_terminal_states[:, 1] + actions - 1, 0, size - 1)

    correct_agent_x_prob = np.array([
        belief.sample().model.transition_model(non_terminal_states[i], actions[i])[0][non_terminal_states[i][0] - 1]
        for i in range(sample_size)
    ])

    correct_agent_y_prob = np.array([
        belief.sample().model.transition_model(non_terminal_states[i], actions[i])[1][correct_next_y[i]]
        for i in range(sample_size)
    ])

    non_terminal_obst_bottom_states = np.stack((
        np.random.randint(1, size, sample_size),
        np.random.randint(0, size, size=sample_size),
        np.zeros(sample_size)
    ), axis=1)

    impossible_obst_transition_prob = np.concatenate([
        belief.sample().model.transition_model(
            non_terminal_obst_bottom_states[i],
            actions[i]
        )[2][2:]
        for i in range(sample_size)
    ])

    obs_stay_bottom_prob = np.array([
        belief.sample().model.transition_model(
            non_terminal_obst_bottom_states[i],
            actions[i]
        )[2][0]
        for i in range(sample_size)
    ])

    non_terminal_obst_middle_states = np.stack((
        np.random.randint(1, size, sample_size),
        np.random.randint(0, size, size=sample_size),
        np.zeros(sample_size) + mid
    ), axis=1)

    obst_stay_mid_prob = np.array([
        belief.sample().model.transition_model(
            non_terminal_obst_middle_states[i],
            actions[i]
        )[2][mid]
        for i in range(sample_size)
    ])

    return [
        ('obstacle-stay-mid-prob', obst_stay_mid_prob),
        ('obstacle-stay-bottom-prob', obs_stay_bottom_prob),
        ('impossible-obstacle-transition-prob', impossible_obst_transition_prob),
        ('agent-x-transition-prob', correct_agent_x_prob),
        ('agent-y-transition-prob', correct_agent_y_prob)
    ]


def ca_observation_analysis(
        belief: ParticleFilter,
        size: int) -> List[Tuple[str, Union[float, np.ndarray]]]:
    """ provides analysis on belief over collision avoidance observation model

    returns the probability of
        - observing obstacle correctly (mid, bottom)
        - observing the x position correctly (deterministic)
        - observing the y position correctly (deterministic)

    Args:
         size: (`int`):
         belief: (`ParticleFilter`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """

    mid = int(size / 2)
    sample_size = 100
    actions = np.random.randint(0, 3, size=sample_size)

    non_terminal_x = np.random.randint(1, size, sample_size)
    y_position = np.random.randint(0, size, size=sample_size)
    next_y_position = np.clip(y_position + actions - 1, 0, size - 1)

    obs = np.random.randint(0, size, size=sample_size)

    non_terminal_states = np.stack((non_terminal_x, y_position, obs), axis=1)
    next_states = np.stack((non_terminal_x - 1, next_y_position, obs), axis=1)

    correct_x_pos_prob = np.array([
        belief.sample().model.observation_model(non_terminal_states[i], actions[i], next_states[i])[0][next_states[i][0]]
        for i in range(sample_size)
    ])

    correct_y_pos_prob = np.array([
        belief.sample().model.observation_model(non_terminal_states[i], actions[i], next_states[i])[1][next_states[i][1]]
        for i in range(sample_size)
    ])

    obs = np.zeros(sample_size)
    non_terminal_states = np.stack((non_terminal_x, y_position, obs), axis=1)
    next_states = np.stack((non_terminal_x - 1, next_y_position, obs), axis=1)

    correct_bottom_obstacle_obs_prob = np.array([
        belief.sample().model.observation_model(
            non_terminal_states[i],
            actions[i], next_states[i]
        )[2][0]
        for i in range(sample_size)
    ])

    obs += mid
    non_terminal_states = np.stack((non_terminal_x, y_position, obs), axis=1)
    next_states = np.stack((non_terminal_x - 1, next_y_position, obs), axis=1)

    correct_mid_obstacle_obs_prob = np.array([
        belief.sample().model.observation_model(
            non_terminal_states[i],
            actions[i], next_states[i]
        )[2][mid]
        for i in range(sample_size)
    ])

    return [
        ('correct-bottom-obstacle-obs-prob', correct_bottom_obstacle_obs_prob),
        ('correct-mid-obstacle-obs-prob', correct_mid_obstacle_obs_prob),
        ('correct-x-obs-prob', correct_x_pos_prob),
        ('correct-y-obs-prob', correct_y_pos_prob),
    ]


def rr_observation_analysis(
        belief: ParticleFilter,
        num_lanes: int) -> List[Tuple[str, Union[float, np.ndarray]]]:
    """ checks whether the deterministic observation function is learned properly

    Args:
         belief: (`ParticleFilter`):
         num_lanes: (`int`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """
    length = RoadRacer.LANE_LENGTH
    size = 500

    # sample states and actions
    states = np.concatenate((
        np.random.randint(0, length, (size, num_lanes)),
        np.random.randint(0, num_lanes, (size, 1))
    ), axis=1)
    actions = np.random.randint(low=0, high=3, size=size)
    models = [belief.sample().model for _ in range(size)]

    next_states = np.array([
        models[i].sample_state(states[i], actions[i])
        for i in range(size)
    ])

    correct_observation_prob = np.array([
        models[i].observation_model(
            states[i],
            actions[i],
            next_states[i])[0][next_states[i][RoadRacer.get_current_lane(next_states[i])]]
        for i in range(size)
    ])

    return [('correct_observation_prob', correct_observation_prob)]


def rr_agent_lane_change(
        belief: ParticleFilter,
        num_lanes: int) -> List[Tuple[str, Union[float, np.ndarray]]]:
    """ diagnoses the model on predicting lane change

    Args:
         belief: (`ParticleFilter`):
         num_lanes: (`int`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """
    length = RoadRacer.LANE_LENGTH
    size = 500

    # sample states and actions
    states = np.concatenate((
        np.random.randint(0, length, (size, num_lanes)),
        np.random.randint(0, num_lanes, (size, 1))
    ), axis=1)
    actions = np.random.randint(low=0, high=3, size=size)
    correct_next_lane = np.clip(a=states[:, -1] + actions - 1, a_min=0, a_max=num_lanes - 1)

    correct_lane_prob = np.array([
        belief.sample().model.transition_model(
            states[i],
            actions[i])[-1][correct_next_lane[i]]
        for i in range(size)
    ])

    return [('correct_agent_y_prob', correct_lane_prob)]


def rr_lane_advances(
        belief: ParticleFilter,
        num_lanes: int) -> List[Tuple[str, Union[float, np.ndarray]]]:
    """ diagnoses the model on predicting lane change

    Args:
         belief: (`ParticleFilter`):
         num_lanes: (`int`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """
    length = RoadRacer.LANE_LENGTH
    size = 500

    # sample states and actions
    states = np.concatenate((
        np.random.randint(0, length, (size, num_lanes)),
        np.random.randint(0, num_lanes, (size, 1))
    ), axis=1)
    actions = np.random.randint(low=0, high=3, size=size)

    lane_prob = np.array([
        belief.sample().model.transition_model(
            states[i],
            actions[i])[:-1]
        for i in range(size)
    ])

    potential_next_lane = np.zeros((size, num_lanes, length), dtype=bool)
    # HATE THIS but do not know how to do this indexing wise...
    for i in range(size):
        potential_next_lane[i, np.arange(num_lanes), states[i, :-1]] = True
        potential_next_lane[i, np.arange(num_lanes), (states[i, :-1] - 1) % length] = True

    incorrect_lane_prob = lane_prob[~potential_next_lane]
    correct_lane_prob = lane_prob[potential_next_lane].reshape(size, num_lanes, 2)

    # states where agent is blocking car
    states[np.arange(size), states[:, -1]] = 1
    actions = np.ones(size, dtype=int)

    car_block_prob = np.array([
        belief.sample().model.transition_model(
            states[i],
            actions[i])[RoadRacer.get_current_lane(states[i])][1]  # stay
        for i in range(size)
    ])

    # randomly checking if passed cars are starting at start again
    states = np.concatenate((
        np.random.randint(0, length, (size, num_lanes)),
        np.random.randint(0, num_lanes, (size, 1))
    ), axis=1)
    actions = np.random.randint(low=0, high=3, size=size)

    random_lanes = np.random.randint(0, num_lanes, size)
    states[np.arange(size), random_lanes] = 0

    car_reappear_prob = np.array([
        belief.sample().model.transition_model(
            states[i],
            actions[i])[random_lanes[i]]
        for i in range(size)
    ])

    return [
        ('incorrect_lane_advance_prob', incorrect_lane_prob),
        ('correct_car_blocked_prob', car_block_prob),
    ] + [
        (f'advance_prob_lane_{i}', correct_lane_prob[:, i, 0])
        for i in range(num_lanes)
    ] + [
        (f'car_stay_edge_prob_{i}', car_reappear_prob[random_lanes == i, 0])
        for i in range(num_lanes)
    ] + [
        (f'car_reappear_edge_prob_{i}', car_reappear_prob[random_lanes == i, -1])
        for i in range(num_lanes)
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
            partial(ca_observation_analysis, size=domain_size),
            count_unique_models
        ])
    if domain == 'road_racer':
        return chain_analysis([
            partial(rr_observation_analysis, num_lanes=domain_size),
            partial(rr_agent_lane_change, num_lanes=domain_size),
            partial(rr_lane_advances, num_lanes=domain_size),
            count_unique_models
        ])

    # default analysis
    return count_unique_models
