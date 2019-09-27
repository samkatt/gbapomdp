""" Neural networks used as Q functions """

from typing import Optional, List
import abc
import numpy as np
import torch

from torch.nn.modules.loss import _Loss as TorchLoss

from agents.neural_networks import misc, networks
from environments import ActionSpace
from misc import POBNRLogger, Space
from pytorch_api import log_tensorboard, device, tensorboard_logging


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


class RMSELoss(TorchLoss):  # type: ignore
    """ custom RMSELoss criterion in pytorch """

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        """ forward pass, returns loss of rmse(prediction, target)

        Args:
             prediction: (`torch.Tensor`):
             target: (`torch.Tensor`):

        """
        return torch.sqrt(
            torch.nn.functional.mse_loss(prediction, target, reduction=self.reduction)
        )


class QNet(QNetInterface, POBNRLogger):
    """ interface to all Q networks """

    def __init__(
            self,
            action_space: ActionSpace,
            obs_space: Space,
            conf,
            name: str):
        """ constructs a QNet

        Args:
             action_space: (`pobnrl.environments.ActionSpace`): output size of the network
             obs_space: (`pobnrl.misc.Space`): of eenvironments
             name: (`str`): name of the network (used for scoping)
             conf: (`namespace`): configurations
             name: (`str`): name of the QNet

        """

        POBNRLogger.__init__(self)

        assert conf.history_len > 0, f'invalid history len ({conf.history_len}) < 1'
        assert conf.batch_size > 0, f'invalid batch size ({conf.batch_size}) < 1'
        assert 1 >= conf.gamma > 0, f'invalid gamma ({conf.gamma})'
        assert conf.learning_rate < .2, f'invalid learning rate ({conf.learning_rate}) < .2'

        self.name = name
        self.history_len = conf.history_len
        self.batch_size = conf.batch_size
        self.gamma = conf.gamma

        self.replay_buffer = misc.ReplayBuffer()

        self.net = networks.Net(
            obs_space.ndim * self.history_len,
            int(action_space.n),
            conf.network_size,
            conf.prior_function_scale
        ).to(device())

        self.target_net = networks.Net(
            obs_space.ndim * self.history_len,
            int(action_space.n),
            conf.network_size,
            conf.prior_function_scale
        ).to(device())

        self.update_target()  # make sure paramters are set equal

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=conf.learning_rate)
        self.criterion = RMSELoss()

        self.num_batches = 0

    def reset(self) -> None:
        """ resets the network and buffer """

        self.replay_buffer.clear()
        self.num_batches = 0

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
            qvals = self.net(torch.from_numpy(obs).view(1, -1).float().to(device())).cpu().numpy()

            if self.log_is_on(POBNRLogger.LogLevel.V4):
                self.log(POBNRLogger.LogLevel.V4, f"DQN: {obs} returned Q: {qvals}")

            return qvals

    def batch_update(self) -> None:
        """ performs a batch update """

        if self.replay_buffer.size < self.batch_size:
            self.log(POBNRLogger.LogLevel.V2, "cannot batch update due to small buf")
            return

        batch = self.replay_buffer.sample(self.batch_size, self.history_len)

        # XXX: currently large part of the computation
        # may consider storing this, instead of fishing from batch..?
        obs_shape = batch[0][0]['obs'].shape
        seq_lengths = np.array([len(trace) for trace in batch])

        # initiate all with padding values
        reward = torch.zeros(self.batch_size, device=device())
        terminal = torch.zeros(self.batch_size, dtype=torch.bool, device=device())
        action = torch.zeros(self.batch_size, dtype=torch.long, device=device())
        obs = torch.zeros((self.batch_size, self.history_len) + obs_shape, dtype=torch.float, device=device())
        next_ob = torch.zeros((self.batch_size, *obs_shape), dtype=torch.float, device=device())

        for i, seq in enumerate(batch):
            obs[i][:seq_lengths[i]] = torch.as_tensor([step['obs'] for step in seq])
            reward[i] = seq[-1]['reward']
            terminal[i] = seq[-1]['terminal']
            action[i] = seq[-1]['action'].item()
            next_ob[i] = torch.from_numpy(seq[-1]['next_obs'])

        # due to padding on the left side, we can simply append the *1*
        # next  observation to the original sequence and remove the first
        next_obs = torch.cat(
            (
                obs[:, 1:],
                torch.zeros((self.batch_size, 1) + obs_shape, device=device())
            ), dim=1)
        next_obs[:, -1] = next_ob

        q_values = self.net(obs.view(self.batch_size, -1)).gather(1, action.unsqueeze(1)).squeeze()

        target_values = reward + self.gamma * torch.where(
            terminal,
            torch.zeros(1, device=device()),
            self.target_net(next_obs.view(self.batch_size, -1)).max(1)[0]
        )

        loss = self.criterion(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if tensorboard_logging():
            log_tensorboard(f'qloss/{self.name}', loss.item(), self.num_batches)
            log_tensorboard(f'q-vals/{self.name}', q_values.view(-1), self.num_batches)
        self.num_batches += 1

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
            conf,
            name: str):
        """ constructs a recurrent q-network

        Args:
             action_space: (`pobnrl.environments.ActionSpace`): output size of the network
             obs_space: (`pobnrl.misc.Space`): of eenvironments
             name: (`str`): name of the network (used for scoping)
             conf: (`namespace`): configurations
             name: (`str`): name of the QNet

        """

        assert conf.history_len > 0, f'invalid history len ({conf.history_len}) < 1'
        assert conf.batch_size > 0, f'invalid batch size ({conf.batch_size}) < 1'
        assert 1 >= conf.gamma > 0, f'invalid gamma ({conf.gamma})'
        assert conf.learning_rate < .2, f'invalid learning rate ({conf.learning_rate}) < .2'

        POBNRLogger.__init__(self)

        self.name = name
        self.history_len = conf.history_len
        self.batch_size = conf.batch_size
        self.gamma = conf.gamma

        self.replay_buffer = misc.ReplayBuffer()
        self.rnn_state = None

        self.net = networks.RecNet(
            obs_space.ndim,
            int(action_space.n),
            conf.network_size,
            conf.prior_function_scale
        ).to(device())

        self.target_net = networks.RecNet(
            obs_space.ndim,
            int(action_space.n),
            conf.network_size,
            conf.prior_function_scale
        ).to(device())

        self.update_target()  # make sure paramters are set equal

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=conf.learning_rate)
        self.criterion = RMSELoss()

        self.num_batches = 0

    def reset(self) -> None:
        """ resets the net internal state, network parameters, and replay buffer """

        self.replay_buffer.clear()
        self.num_batches = 0
        self.rnn_state = None

        self.net.random_init_parameters()
        self.update_target()  # set target equal to net

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

            if self.log_is_on(POBNRLogger.LogLevel.V4):
                self.log(
                    POBNRLogger.LogLevel.V4,
                    f"DRQN with obs {obs} "
                    f"and state {self.rnn_to_str(self.rnn_state)}"
                )

            qvals, self.rnn_state = self.net(
                torch.from_numpy(obs[-1]).view(1, 1, -1).to(device()).float(),
                self.rnn_state
            )

            qvals = qvals.squeeze().cpu().numpy()

            if self.log_is_on(POBNRLogger.LogLevel.V4):
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

        # XXX: currently large part of the computation
        # may consider storing this, instead of fishing from batch..?
        obs_shape = batch[0][0]['obs'].shape
        seq_lengths = np.array([len(trace) for trace in batch])

        # initiate all with padding values
        reward = torch.zeros(self.batch_size, device=device())
        terminal = torch.zeros(self.batch_size, dtype=torch.bool, device=device())
        action = torch.zeros(self.batch_size, dtype=torch.long, device=device())
        obs = torch.zeros((self.batch_size, self.history_len) + obs_shape, dtype=torch.float, device=device())
        next_ob = torch.zeros((self.batch_size, *obs_shape), dtype=torch.float, device=device())

        for i, seq in enumerate(batch):
            obs[i][:seq_lengths[i]] = torch.as_tensor([step['obs'] for step in seq])
            reward[i] = seq[-1]['reward']
            terminal[i] = seq[-1]['terminal']
            action[i] = seq[-1]['action'].item()
            next_ob[i] = torch.from_numpy(seq[-1]['next_obs'])

        # next_obs are being appened where the sequence ended
        next_obs = torch.cat((obs[:, 1:], torch.zeros((self.batch_size, 1) + obs_shape, device=device())), dim=1)
        next_obs[np.arange(self.batch_size), seq_lengths - 1] = next_ob

        # BxhxA
        q_values, _ = self.net(obs.view(self.batch_size, self.history_len, -1))
        # BxA
        q_values = q_values[np.arange(self.batch_size), seq_lengths - 1]
        # B
        q_values = q_values.gather(1, action.unsqueeze(1)).squeeze()

        # BxhxA
        expected_return, _ = self.target_net(next_obs.view(self.batch_size, self.history_len, -1))
        # B
        expected_return = expected_return[np.arange(self.batch_size), seq_lengths - 1].max(1)[0]

        target_values = reward + self.gamma * torch.where(
            terminal,
            torch.zeros(1, device=device()),
            expected_return
        )

        loss = self.criterion(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if tensorboard_logging():
            log_tensorboard(f'qloss/{self.name}', loss.item(), self.num_batches)
            log_tensorboard(f'q-vals/{self.name}', q_values.view(-1), self.num_batches)
        self.num_batches += 1

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
            hidden_state_descr = 'None' if not rnn_state[0] else str(rnn_state[0][0][0][0][0])
            prior_descr = 'None' if not rnn_state[1] else str(rnn_state[1][0][0][0][0])

            return f'hidden state: {hidden_state_descr}, prior state: {prior_descr}'

        return str(rnn_state)
