""" domains either learned or constructed from other domains """

from collections import namedtuple
from typing import Callable
import numpy as np
import tensorflow as tf

from agents.neural_networks import ReplayBuffer
from agents.neural_networks.neural_pomdps import DynamicsModel
from environments import ActionSpace
from environments import POUCTSimulator, POUCTInteraction
from misc import DiscreteSpace


def generate_replay_buffer(domain: POUCTSimulator) -> ReplayBuffer:

    replay_buffer = ReplayBuffer()

    for _ in range(replay_buffer.capacity):

        terminal = False
        state = domain.sample_start_state()
        while not terminal:

            action = domain.action_space.sample()

            step = domain.simulation_step(state, action)
            terminal = step.terminal

            replay_buffer.store(
                (state.copy(), action, step.state.copy(), step.observation.copy()),
                terminal
            )

            state = step.state

    assert replay_buffer.size == replay_buffer.capacity
    return replay_buffer


class PretrainedNeuralPOMDP(POUCTSimulator):

    class AugmentedState(namedtuple('bn_pomdp_state', 'domain_state model')):

        __slots__ = ()  # required to keep lightweight implementation of namedtuple

    def __init__(
            self,
            domain: POUCTSimulator,
            state_space: DiscreteSpace,
            reward_function: Callable[[np.ndarray, int, np.ndarray], float],
            terminal_checker: Callable[[np.ndarray, int, np.ndarray], bool],
            name: str):

        # domain knowledge
        self.domain_obs2index = domain.obs2index
        self.sample_domain_start_state = domain.sample_start_state
        self.domain_reward = reward_function
        self.domain_terminal = terminal_checker
        self.domain_action_space = domain.action_space
        self.domain_obs_space = domain.observation_space

        replay_buffer = generate_replay_buffer(domain)

        self._models = [
            DynamicsModel(
                state_space,
                domain.action_space,
                domain.observation_space,
                64,  # TODO: allow conf input
                tf.train.AdamOptimizer(),  # TODO: allow conf input
                f"{name}_model_{i}"
            ) for i in range(10)
        ]

        for model in self._models:
            for _ in range(100):  # TOOD: get from conf

                batch = replay_buffer.sample(batch_size=32)

                # grab the elements from the batch
                states = np.array([seq[0][0] for seq in batch])
                actions = np.array([seq[0][1] for seq in batch])
                new_states = np.array([seq[0][2] for seq in batch])
                observations = np.array([seq[0][3] for seq in batch])

                model.batch_update(states, actions, observations, new_states)

    def sample_start_state(self) -> 'AugmentedState':
        """ returns a sample initial (internal) state and some neural network """
        return self.AugmentedState(
            self.sample_domain_start_state(),
            np.random.choice(self._models)
        )

    def simulation_step(
            self,
            state: 'AugmentedState',
            action: int) -> POUCTInteraction:

        # use model to generate a step
        new_domain_state, obs = state.model.simulation_step(state.domain_state, action)

        # domain knowledge to get reward and terminality
        rew = self.domain_reward(state.domain_state, action, new_domain_state)
        term = self.domain_terminal(state.domain_state, action, new_domain_state)

        return POUCTInteraction(
            self.AugmentedState(new_domain_state, state.model), obs, rew, term
        )

    def obs2index(self, observation: np.ndarray) -> int:
        return self.domain_obs2index(observation)

    def action_space(self) -> ActionSpace:
        return self.domain_action_space

    def observation_space(self) -> DiscreteSpace:
        return self.domain_obs_space
