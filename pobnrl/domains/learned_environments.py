""" domains either learned or constructed from other domains """

from collections import namedtuple
from typing import Callable
import numpy as np

from agents.neural_networks import ReplayBuffer
from agents.neural_networks.neural_pomdps import DynamicsModel
from environments import ActionSpace
from environments import POUCTSimulator, POUCTInteraction
from misc import DiscreteSpace


def generate_replay_buffer(domain: POUCTSimulator) -> ReplayBuffer:
    """ Fills up a replay buffer of (s,a,s',o) interactions

    Args:
         domain: (`pobnrl.environments.POUCTSimulator`): a simulator to generate interactions

    RETURNS (`pobnrl.agents.neural_networks.misc.ReplayBuffer`):

    """

    replay_buffer = ReplayBuffer()

    for _ in range(replay_buffer.capacity):

        terminal = False
        state = domain.sample_start_state()
        while not terminal:

            action = domain.action_space.sample()

            step = domain.simulation_step(state, action)
            terminal = step.terminal

            replay_buffer.store(
                (state.copy(), action, step.state.copy(), step.observation),
                terminal
            )

            state = step.state

    assert replay_buffer.size == replay_buffer.capacity
    return replay_buffer


class NeuralEnsemble(POUCTSimulator):  # pylint: disable=too-many-instance-attributes
    """ A simulator over (`pobnrl.agents.neural_networks.neural_pomdps.DynamicsModel`, state) states """

    class AugmentedState(namedtuple('bn_pomdp_state', 'domain_state model')):
        """ A state containing (POMDP state, POMDP dynamics) """

        # required to keep lightweight implementation of namedtuple
        __slots__ = ()

    def __init__(  # pylint: disable=too-many-arguments
            self,
            domain: POUCTSimulator,
            state_space: DiscreteSpace,
            reward_function: Callable[[np.ndarray, int, np.ndarray], float],
            terminal_checker: Callable[[np.ndarray, int, np.ndarray], bool],
            conf,
            name: str):
        """ Creates `NeuralEnsemble`

        Args:
             domain: (`pobnrl.environments.POUCTSimulator`): domain to train interactions from
             state_space: (`pobnrl.misc.DiscreteSpace`): the state space of the `domain`
             reward_function: (`Callable[[np.ndarray, int, np.ndarray], float]`): reward function
             terminal_checker: (`Callable[[np.ndarray, int, np.ndarray], bool]`): termination check
             conf: configurations from program input (network_size and learning_rate)
             name: (`str`): name (unique) of this simulator

        """

        # settings
        self._batch_size = conf.batch_size

        # domain knowledge
        self.domain_obs2index = domain.obs2index
        self.sample_domain_start_state = domain.sample_start_state
        self.domain_action_space = domain.action_space
        self.domain_obs_space = domain.observation_space

        self.domain_reward = reward_function
        self.domain_terminal = terminal_checker

        self._models = [
            DynamicsModel(
                state_space,
                domain.action_space,
                domain.observation_space,
                conf,
                f"{name}_model_{i}"
            ) for i in range(conf.num_nets)
        ]

    def learn_dynamics_offline(self, simulator: POUCTSimulator, num_epochs: int):
        """  learn the dynamics function offline, given simulator

        Args:
             simulator: (`pobnrl.environments.POUCTSimulator`): simulator to generate interactions from
             num_epochs: (`int`): number of batch updates

        """

        for model in self._models:

            replay_buffer = generate_replay_buffer(simulator)

            for _ in range(num_epochs):

                batch = replay_buffer.sample(batch_size=self._batch_size)

                # grab the elements from the batch
                states = np.array([seq[0][0] for seq in batch])
                actions = np.array([seq[0][1] for seq in batch])
                new_states = np.array([seq[0][2] for seq in batch])
                observations = np.array([seq[0][3] for seq in batch])

                model.batch_update(states, actions, new_states, observations)

    def sample_start_state(self) -> 'AugmentedState':
        """  returns a sample initial (internal) state and some neural network

        Args:

        RETURNS (`AugmentedState`):

        """
        return self.AugmentedState(
            self.sample_domain_start_state(),
            np.random.choice(self._models)
        )

    def simulation_step(
            self,
            state: AugmentedState,
            action: int) -> POUCTInteraction:
        """ Performs simulation step

        Args:
             state: (`AugmentedState`): incoming state
             action: (`int`): action

        RETURNS (`pobnrl.environments.POUCTInteraction`):

        """

        # use model to generate a step
        new_domain_state, obs = state.model.simulation_step(state.domain_state, action)

        # domain knowledge to get reward and terminality
        rew = self.domain_reward(state.domain_state, action, new_domain_state)
        term = self.domain_terminal(state.domain_state, action, new_domain_state)

        return POUCTInteraction(
            self.AugmentedState(new_domain_state, state.model), obs, rew, term
        )

    def obs2index(self, observation: np.ndarray) -> int:
        """ projects observation to single dimension (scalar)

        Args:
             observation: (`np.ndarray`):

        RETURNS (`int`):

        """
        return self.domain_obs2index(observation)

    @property
    def action_space(self) -> ActionSpace:
        """ returns `this` actions space

        Args:

        RETURNS (`pobnrl.environments.ActionSpace`):

        """
        return self.domain_action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ returns `this` observation space

        Args:

        RETURNS (`pobnrl.misc.DiscreteSpace`):

        """
        return self.domain_obs_space
