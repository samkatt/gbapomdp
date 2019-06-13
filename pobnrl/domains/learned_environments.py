""" domains either learned or constructed from other domains """

from typing import Callable
import numpy as np

from agents.neural_networks import ReplayBuffer
from agents.neural_networks.neural_pomdps import DynamicsModel
from environments import ActionSpace
from environments import Simulator, SimulationResult
from misc import DiscreteSpace, POBNRLogger


def generate_replay_buffer(domain: Simulator) -> ReplayBuffer:
    """ Fills up a replay buffer of (s,a,s',o) interactions

    Args:
         domain: (`pobnrl.environments.Simulator`): a simulator to generate interactions

    RETURNS (`pobnrl.agents.neural_networks.misc.ReplayBuffer`):

    """

    replay_buffer = ReplayBuffer()

    for _ in range(replay_buffer.capacity):

        terminal = False
        state = domain.sample_start_state()
        while not terminal:

            action = domain.action_space.sample()

            step = domain.simulation_step(state, action)
            terminal = domain.terminal(state, action, step.state)

            replay_buffer.store(
                (state.copy(), action, step.state.copy(), step.observation),
                terminal
            )

            state = step.state

    assert replay_buffer.size == replay_buffer.capacity
    return replay_buffer


class NeuralEnsemble(Simulator, POBNRLogger):
    """ A simulator over (`pobnrl.agents.neural_networks.neural_pomdps.DynamicsModel`, state) states """

    class AugmentedState:
        """ A state containing (POMDP state, POMDP dynamics) """

        def __init__(self, domain_state: np.ndarray, model: DynamicsModel):

            self.domain_state = domain_state
            self.model = model

        def __repr__(self) -> str:
            return f'Augmented state: state {self.domain_state} with model {self.model}'

    def __init__(self, domain: Simulator, conf, name: str):
        """ Creates `NeuralEnsemble`

        Args:
             domain: (`pobnrl.environments.Simulator`): domain to train interactions from
             conf: configurations from program input (`network_size` and `learning_rate`)
             name: (`str`): name (unique) of this simulator

        """

        POBNRLogger.__init__(self)

        # settings
        self._batch_size = conf.batch_size

        # domain knowledge
        self.domain_action_space = domain.action_space
        self.domain_obs_space = domain.observation_space

        self.sample_domain_start_state = domain.sample_start_state
        self.domain_obs2index = domain.obs2index

        self.domain_reward = domain.reward
        self.domain_terminal = domain.terminal

        self._models = [
            DynamicsModel(
                domain.state_space,
                domain.action_space,
                domain.observation_space,
                conf,
                f"{name}_model_{i}"
            ) for i in range(conf.num_nets)
        ]

    def learn_dynamics_offline(
            self,
            simulator_sampler: Callable[[], Simulator],
            num_epochs: int):
        """  learn the dynamics function offline, given simulator

        If simulator_sampler returns the true environment (all time), this can
        be considered 'training on true environment'. If the sampler samples
        from some distribution, then we are 'training on prior'

        Args:
             simulator_sampler: (`Callable[[], ` `pobnrl.environments.Simulator` `]`): function to sample environments
             num_epochs: (`int`): number of batch updates

        """

        for i, model in enumerate(self._models):

            sim = simulator_sampler()
            replay_buffer = generate_replay_buffer(sim)

            self.log(
                POBNRLogger.LogLevel.V1,
                f'Learning offline: learning net {i+1}/{len(self._models)} on {sim}'
            )

            for _ in range(num_epochs):

                batch = replay_buffer.sample(batch_size=self._batch_size)

                # grab the elements from the batch
                states = np.array([seq[0][0] for seq in batch])
                actions = np.array([seq[0][1] for seq in batch])
                new_states = np.array([seq[0][2] for seq in batch])
                observations = np.array([seq[0][3] for seq in batch])

                model.batch_update(states, actions, new_states, observations)

    @property
    def state_space(self) -> DiscreteSpace:
        """ No method should want to know `this` state space

        Args:

        RETURNS (`pobnrl.misc.DiscreteSpace`):

        """
        raise NotImplementedError

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
            action: int) -> SimulationResult:
        """ Performs simulation step

        Args:
             state: (`AugmentedState`): incoming state
             action: (`int`): action

        RETURNS (`pobnrl.environments.SimulationResult`):

        """

        # use model to generate a step
        new_domain_state, obs = state.model.simulation_step(state.domain_state, action)

        return SimulationResult(self.AugmentedState(new_domain_state, state.model), obs)

    def reward(self, state: AugmentedState, action: int, new_state: AugmentedState) -> float:
        """ the reward function of the underlying environment

        Args:
             state: (`AugmentedState`):
             action: (`int`):
             new_state: (`AugmentedState`):

        RETURNS (`float`): the reward of the transition

        """
        return self.domain_reward(state.domain_state, action, new_state.domain_state)

    def terminal(self, state: AugmentedState, action: int, new_state: AugmentedState) -> bool:
        """ the termination function of the underlying environment

        Args:
             state: (`AugmentedState`):
             action: (`int`):
             new_state: (`AugmentedState`):

        RETURNS (`bool`): whether the transition is terminal

        """
        return self.domain_terminal(state.domain_state, action, new_state.domain_state)

    def obs2index(self, observation: np.ndarray) -> int:
        """ projects observation to single dimension (scalar)

        Args:
             observation: (`np.ndarray`):

        RETURNS (`int`):

        """
        return self.domain_obs2index(observation)
