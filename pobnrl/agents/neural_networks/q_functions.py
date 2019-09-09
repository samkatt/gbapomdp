""" Neural networks used as Q functions """

from typing import Optional, List
import abc
import copy
import numpy as np
import torch

from agents.neural_networks import misc, networks
from environments import ActionSpace
from misc import POBNRLogger, Space


class QNetInterface(abc.ABC):
    """ interface to all Q networks """

    @abc.abstractmethod
    def reset(self) -> None:
        """ resets to initial state """

    @abc.abstractmethod
    def episode_reset(self) -> None:
        """ resets the internal state to prepare for a new episode """

    @abc.abstractmethod
    def qvalues(self, obs: np.ndarray) -> np.array:
        """ returns the Q-values associated with the obs

        Args:
             obs: (`np.ndarray`): the net input

        RETURNS (`np.array`): q-value for each action

        """

    @abc.abstractmethod
    def batch_update(self) -> None:
        """ performs a batch update """

    @abc.abstractmethod
    def update_target(self) -> None:
        """ updates the target network """

    @abc.abstractmethod
    def record_transition(
            self,
            observation: np.ndarray,
            action: int,
            reward: float,
            next_observation: np.ndarray,
            terminal: bool) -> None:
        """ notifies this of provided transition

        Args:
             observation: (`np.ndarray`): shape depends on env
             action: (`int`): the taken action
             reward: (`float`): the resulting reward
             next_observation: (`np.ndarray`): shape depends on env
             terminal: (`bool`): whether transition was terminal
        """


class QNet(QNetInterface, POBNRLogger):
    """ interface to all Q networks """

    def __init__(
            self,
            action_space: ActionSpace,
            obs_space: Space,
            conf):

        POBNRLogger.__init__(self)

        assert conf.history_len > 0, f'invalid history len ({conf.history_len}) < 1'
        assert conf.batch_size > 0, f'invalid batch size ({conf.batch_size}) < 1'
        assert 1 >= conf.gamma > 0, f'invalid gamma ({conf.gamma})'
        assert conf.learning_rate < .2, f'invalid learning rate ({conf.learning_rate}) < .2'

        self.history_len = conf.history_len
        self.batch_size = conf.batch_size
        self.gamma = conf.gamma

        self.replay_buffer = misc.ReplayBuffer()

        self.net = networks.Net(
            obs_space.ndim * self.history_len,
            action_space.n,
            conf.network_size
        )
        self.net.random_init_parameters()

        self.target_net = copy.deepcopy(self.net)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=conf.learning_rate)
        self.criterion = misc.loss_criterion(conf.loss)

    def reset(self) -> None:
        """ resets the network and buffer """

        self.replay_buffer.clear()

        self.net.random_init_parameters()
        self.update_target()  # set target equal to net

    def episode_reset(self) -> None:
        """ empty """

    def qvalues(self, obs: np.ndarray) -> np.ndarray:
        """ returns the Q-values associated with the obs

        Args:
             obs: (`np.ndarray`): the net input

        RETURNS (`np.ndarray`): q-value for each action

        """

        assert obs.ndim >= 2, \
            f'observation ({obs.ndim}) should be history len by obs size'
        assert obs.shape[0] <= self.history_len, \
            f'first dimension of observation ({obs.shape[0]}) must be < {self.history_len}'

        # pad observation if necessary

        if not len(obs) == self.history_len:
            padding = [(0, 0) for _ in range(len(obs[0].shape) + 1)]
            padding[0] = (self.history_len - len(obs), 0)

            obs = np.pad(obs, padding, 'constant')

        with torch.no_grad():
            qvals = self.net(torch.from_numpy(obs).reshape(1, -1).float()).numpy()
            self.log(POBNRLogger.LogLevel.V4, f"DQN: {obs} returned Q: {qvals}")
            return qvals

    def batch_update(self) -> None:
        """ performs a batch update """

        if self.replay_buffer.size < self.batch_size:
            self.log(POBNRLogger.LogLevel.V2, "cannot batch update due to small buf")
            return

        batch = self.replay_buffer.sample(self.batch_size, self.history_len)

        # TODO: this is ugly and must be improved on
        # may consider storing this, instead of fishing from batch..?
        obs_shape = batch[0][0]['obs'].shape
        seq_lengths = np.array([len(trace) for trace in batch])

        # initiate all with padding values
        reward = torch.zeros(self.batch_size)
        terminal = torch.zeros(self.batch_size, dtype=torch.bool)
        action = torch.zeros(self.batch_size, dtype=torch.long)
        obs = torch.zeros((self.batch_size, self.history_len) + obs_shape)
        next_ob = torch.zeros((self.batch_size, *obs_shape))

        for i, seq in enumerate(batch):
            obs[i][:seq_lengths[i]] = torch.as_tensor([step['obs'] for step in seq], dtype=torch.int)
            reward[i] = seq[-1]['reward']
            terminal[i] = seq[-1]['terminal']
            action[i] = seq[-1]['action'].item()
            next_ob[i] = torch.from_numpy(seq[-1]['next_obs'])

        # due to padding on the left side, we can simply append the *1*
        # next  observation to the original sequence and remove the first
        # TODO: torch.cat?
        next_obs = torch.from_numpy(np.concatenate((obs[:, 1:], np.zeros((self.batch_size, 1) + obs_shape)), axis=1)).float()
        next_obs[:, -1] = next_ob

        q_values = self.net(obs.reshape(self.batch_size, -1)).gather(1, action.unsqueeze(1)).squeeze()

        target_values = reward + self.gamma * torch.where(
            terminal,
            torch.zeros(self.batch_size),
            self.target_net(next_obs.reshape(self.batch_size, -1)).max(1)[0]
        )
        loss = self.criterion(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self) -> None:
        """ updates the target network """
        self.target_net.load_state_dict(self.net.state_dict())

    def record_transition(
            self,
            observation: np.ndarray,
            action: int,
            reward: float,
            next_observation: np.ndarray,
            terminal: bool) -> None:
        """ notifies this of provided transition

        Args:
             observation: (`np.ndarray`): shape depends on env
             action: (`int`): the taken action
             reward: (`float`): the resulting reward
             next_observation: (`np.ndarray`): shape depends on env
             terminal: (`bool`): whether transition was terminal
        """

        self.replay_buffer.store(
            {
                'obs': observation,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'next_obs': next_observation
            },
            terminal
        )


class RecQNet(QNetInterface, POBNRLogger):
    """ a network based on DRQN that can return q values and update """

    def __init__(
            self,
            action_space: ActionSpace,
            obs_space: Space,
            conf):
        """ construct the DRQNNet

        Assumes the rec_q_func provided is a recurrent Q function

        Args:
             action_space: (`pobnrl.environments.ActionSpace`): output size of the network
             obs_space: (`pobnrl.misc.Space`): of eenvironments
             rec_q_func: (`Callable`): the (recurrent) Q function
             optimizer: the tf.optimizer optimizer to use for learning
             name: (`str`): name of the network (used for scoping)
             conf: (`namespace`): configurations

        """

        assert conf.history_len > 0, f'invalid history len ({conf.history_len}) < 1'
        assert conf.batch_size > 0, f'invalid batch size ({conf.batch_size}) < 1'
        assert 1 >= conf.gamma > 0, f'invalid gamma ({conf.gamma})'
        assert conf.learning_rate < .2, f'invalid learning rate ({conf.learning_rate}) < .2'

        POBNRLogger.__init__(self)

        self.history_len = conf.history_len
        self.batch_size = conf.batch_size
        self.gamma = conf.gamma

        self.replay_buffer = misc.ReplayBuffer()
        self.rnn_state = None

        self.net = networks.RecNet(
            obs_space.ndim,
            action_space.n,
            conf.network_size
        )

        self.net.random_init_parameters()

        self.target_net = copy.deepcopy(self.net)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=conf.learning_rate)
        self.criterion = misc.loss_criterion(conf.loss)

    def reset(self) -> None:
        """ resets the net internal state, network parameters, and replay buffer """

        self.replay_buffer.clear()
        self.rnn_state = None

        self.net.random_init_parameters()
        self.update_target()

    def episode_reset(self) -> None:
        """ resets the net internal state """
        self.rnn_state = None

    def qvalues(self, obs: np.ndarray) -> np.array:
        """ returns the Q-values associated with the obs

        Only actually supplies the network with the last observation because
        of the recurrent part of the network

        Args:
             obs: (`np.ndarray`): the net input

        RETURNS (`np.array`): q-value for each action

        """

        assert obs.ndim >= 2, \
            f'observation ({obs.ndim}) should be history len by obs size'
        assert obs.shape[0] <= self.history_len, \
            f'first dimension of observation ({obs.shape[0]}) must be < {self.history_len}'

        with torch.no_grad():

            self.log(
                POBNRLogger.LogLevel.V4,
                f"DRQN with obs {obs} "
                f"and state {self.rnn_to_str(self.rnn_state)}"
            )

            qvals, self.rnn_state = self.net(
                torch.from_numpy(obs[-1]).reshape(1, 1, -1).float(),
                self.rnn_state
            )

            qvals = qvals.squeeze().numpy()

            self.log(
                POBNRLogger.LogLevel.V4,
                f"DRQN: returned Q: {qvals} "
                f"(first rnn: {self.rnn_to_str(self.rnn_state)})"
            )

        return qvals

    def batch_update(self) -> None:
        """ performs a batch update """

        if self.replay_buffer.size < self.batch_size:
            self.log(
                POBNRLogger.LogLevel.V2,
                f"Network cannot batch update due to small buf"
            )

            return

        batch = self.replay_buffer.sample(self.batch_size, self.history_len)

        # may consider storing this, instead of fishing from batch..?
        obs_shape = batch[0][0]['obs'].shape
        seq_lengths = np.array([len(trace) for trace in batch])

        # initiate all with padding values
        reward = torch.zeros(self.batch_size)
        terminal = torch.zeros(self.batch_size, dtype=torch.bool)
        action = torch.zeros(self.batch_size, dtype=torch.long)
        obs = torch.zeros((self.batch_size, self.history_len) + obs_shape)
        next_ob = torch.zeros((self.batch_size, *obs_shape))

        for i, seq in enumerate(batch):
            obs[i][:seq_lengths[i]] = torch.as_tensor([step['obs'] for step in seq])
            reward[i] = seq[-1]['reward']
            terminal[i] = seq[-1]['terminal']
            action[i] = seq[-1]['action'].item()
            next_ob[i] = torch.from_numpy(seq[-1]['next_obs'])

        # next_obs are being appened where the sequence ended
        next_obs = torch.cat(
            (obs[:, 1:], torch.zeros((self.batch_size, 1) + obs_shape)),
            dim=1,
        )
        next_obs[np.arange(self.batch_size), seq_lengths - 1] = next_ob

        # BxhxA
        q_values, _ = self.net(obs.reshape(self.batch_size, self.history_len, -1))
        # BxA
        q_values = q_values[np.arange(self.batch_size), seq_lengths - 1]
        # B
        q_values = q_values.gather(1, action.unsqueeze(1)).squeeze()

        # BxhxA
        expected_return, _ = self.target_net(next_obs.reshape(self.batch_size, self.history_len, -1))
        # B
        expected_return = expected_return[np.arange(self.batch_size), seq_lengths - 1].max(1)[0]

        target_values = reward + self.gamma * torch.where(
            terminal,
            torch.zeros(self.batch_size),
            expected_return
        )

        loss = self.criterion(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self) -> None:
        """ updates the target network """
        self.target_net.load_state_dict(self.net.state_dict())

    def record_transition(
            self,
            observation: np.ndarray,
            action: int,
            reward: float,
            next_observation: np.ndarray,
            terminal: bool) -> None:
        """ notifies this of provided transition

        Stores transition in replay buffer

        Args:
             observation: (`np.ndarray`): shape depends on env
             action: (`int`): the taken action
             reward: (`float`): the resulting reward
             next_observation: (`np.ndarray`): shape depends on env
             terminal: (`bool`): whether transition was terminal
        """
        self.replay_buffer.store(
            {
                'obs': observation,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'next_obs': next_observation
            },
            terminal
        )

    def rnn_to_str(self, rnn_state: Optional[List[np.ndarray]]) -> str:
        """ returns a string representation of the rnn

        If provided rnn is none, then it will attempt to use the rnn_state of `self`

        Args:
             rnn_state: (`Optional[List[np.ndarray]]`):

        RETURNS (`str`):

        """

        if not rnn_state:
            rnn_state = self.rnn_state

        if rnn_state:
            return f'1st state: {rnn_state[0][0][0][0][0]}, prior: {rnn_state[1][0][0][0][0]}'

        return str(rnn_state)
