""" Neural networks used as Q functions """

from typing import Callable, Optional, List
import abc
import copy
import numpy as np
import tensorflow as tf
import torch

from agents.neural_networks import misc, networks
from agents.neural_networks.utils import update_variables
from environments import ActionSpace
from misc import POBNRLogger, Space
from tf_api import tf_run, tf_board_write, tf_writing_to_board


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
            self.log(
                POBNRLogger.LogLevel.V2,
                f"cannot batch update due to small buf"
            )
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


class DRQNNet(QNetInterface, POBNRLogger):
    """ a network based on DRQN that can return q values and update """

    def __init__(
            self,
            action_space: ActionSpace,
            obs_space: Space,
            rec_q_func: Callable,  # type: ignore
            optimizer,
            name: str,
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

        POBNRLogger.__init__(self)

        self.name = name
        self.history_len = conf.history_len
        self.batch_size = conf.batch_size

        self.replay_buffer = misc.ReplayBuffer()
        self.rnn_state = None

        # shape of network input: variable in first dimension because
        # we sometimes provide complete sequences (for batch updates)
        # and sometimes just the last observation
        input_shape = (None, obs_space.ndim)

        # training operation place holders
        with tf.name_scope(self.name):

            self.obs_ph = tf.compat.v1.placeholder(tf.float32, [None, *input_shape], name="obs")
            self.act_ph = tf.compat.v1.placeholder(tf.int32, [None], name="actions")

            self.next_obs_ph = tf.compat.v1.placeholder(
                tf.float32, [None, *input_shape], name="next_obs"
            )
            self.rew_ph = tf.compat.v1.placeholder(tf.float32, [None], name="rewards")
            self.done_mask_ph = tf.compat.v1.placeholder(tf.bool, [None], name="terminals")

            self.rnn_state_ph_c, self.rnn_state_ph_h\
                = tf.keras.layers.LSTMCell(conf.network_size).get_initial_state(
                    batch_size=tf.shape(self.obs_ph)[0], dtype=tf.float32
                )

            # training operation q values and targets
            with tf.name_scope("net"):

                self.qvalues_fn, self.rec_state_fn = rec_q_func(
                    self.obs_ph,
                    [self.rnn_state_ph_c, self.rnn_state_ph_h],
                    action_space.n,
                    conf.network_size
                )

                net_vars = tf.compat.v1.trainable_variables(scope=tf.compat.v1.get_default_graph().get_name_scope())

                if conf.prior_function_scale:

                    self.log(
                        POBNRLogger.LogLevel.V0,
                        "Prior functions with DRQN currently is **BUGGY**!"
                    )

                    with tf.name_scope("prior_function"):

                        prior_vals, _ = networks.simple_fc_rnn(
                            self.obs_ph,
                            # FIXME: currently the prior rnn state is **NOT**
                            # being maintained. This is a **BUG**
                            tf.keras.layers.LSTMCell(4).get_initial_state(
                                batch_size=tf.shape(self.obs_ph)[0], dtype=tf.float32
                            ),
                            action_space.n,
                            4
                        )

                        scaled_prior = tf.scalar_mul(conf.prior_function_scale, prior_vals)
                        self.qvalues_fn = tf.add(self.qvalues_fn, scaled_prior)

                action_onehot = tf.stack(
                    [tf.range(tf.size(self.act_ph)), self.act_ph], axis=-1
                )

                q_values = tf.gather_nd(self.qvalues_fn, action_onehot, name="pick_Q")

            # target network
            with tf.name_scope("target"):

                next_targets_fn, _ = rec_q_func(
                    self.next_obs_ph,
                    # this network is only used during training
                    # so the initial rnn state will always be 'zero state'
                    tf.keras.layers.LSTMCell(conf.network_size).get_initial_state(
                        batch_size=tf.shape(self.next_obs_ph)[0], dtype=tf.float32
                    ),
                    action_space.n,
                    conf.network_size
                )

                target_vars = tf.compat.v1.trainable_variables(scope=tf.compat.v1.get_default_graph().get_name_scope())

                if conf.prior_function_scale:
                    with tf.name_scope("prior_function"):

                        next_prior_vals, _ = networks.simple_fc_rnn(
                            self.next_obs_ph,
                            # this network is only used during training
                            # so the initial rnn state will always be 'zero state'
                            tf.keras.layers.LSTMCell(4).get_initial_state(
                                batch_size=tf.shape(self.next_obs_ph)[0], dtype=tf.float32
                            ),
                            action_space.n,
                            4
                        )

                        scaled_target_prior = tf.scalar_mul(conf.prior_function_scale, next_prior_vals)
                        next_targets_fn = tf.add(next_targets_fn, scaled_target_prior)

            with tf.name_scope('compute_target'):

                targets = tf.where(
                    self.done_mask_ph,
                    x=self.rew_ph,
                    y=self.rew_ph + (conf.gamma * tf.reduce_max(next_targets_fn, axis=-1))
                )

            loss = misc.loss(q_values, targets, conf.loss)

            gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=net_vars))
            clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, conf.clipping, name='clipping')

            self.train_op = optimizer.apply_gradients(zip(clipped_gradients, variables))

            self.update_target_op = update_variables(target_vars, net_vars)

            if not tf_writing_to_board(conf):
                self.train_diag = tf.no_op('no-diagnostics')

            else:
                loss_summary = tf.compat.v1.summary.scalar('loss', tf.reduce_mean(loss))
                q_values_summary = tf.compat.v1.summary.histogram('q-values', q_values)
                global_norm_summary = tf.compat.v1.summary.scalar('global-norm', global_norm)

                self.train_diag = tf.compat.v1.summary.merge([
                    loss_summary,
                    q_values_summary,
                    global_norm_summary,
                ], name="diagnostics")

    def reset(self) -> None:
        """ resets the net internal state and replay buffer

        Weights are controlled in tensorflow and reset outside of this scope

        """

        self.replay_buffer.clear()
        self.rnn_state = None

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

        assert obs.ndim >= 2, "observation expected to be len x shape"
        assert obs.shape[0] <= self.history_len

        feed_dict = {
            self.obs_ph: obs[-1, None, None]  # cast last ob to shape
        }

        if self.rnn_state is not None:
            feed_dict[self.rnn_state_ph_c], feed_dict[self.rnn_state_ph_h] = self.rnn_state

        self.log(
            POBNRLogger.LogLevel.V4,
            f"DRQN with obs {obs} "
            f"and state {self.rnn_to_str(self.rnn_state)}"
        )

        qvals, self.rnn_state = tf_run(
            [self.qvalues_fn, self.rec_state_fn],
            feed_dict=feed_dict
        )

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
                f"Network {self.name} cannot batch update due to small buf"
            )

            return

        batch = self.replay_buffer.sample(self.batch_size, self.history_len)

        # may consider storing this, instead of fishing from batch..?
        obs_shape = batch[0][0]['obs'].shape
        seq_lengths = np.array([len(trace) for trace in batch])

        # initiate all with padding values
        reward = np.zeros(self.batch_size)
        terminal = np.zeros(self.batch_size).astype(bool)
        action = np.zeros(self.batch_size).astype(int)
        obs = np.zeros((self.batch_size, self.history_len) + obs_shape)
        next_ob = np.zeros((self.batch_size, *obs_shape))

        for i, seq in enumerate(batch):
            obs[i][:seq_lengths[i]] = [step['obs'] for step in seq]
            reward[i] = seq[-1]['reward']
            terminal[i] = seq[-1]['terminal']
            action[i] = seq[-1]['action']
            next_ob[i] = seq[-1]['next_obs']

        # next_obs are being appened where the sequence ended
        next_obs = np.concatenate((obs[:, 1:], np.zeros((self.batch_size, 1) + obs_shape)), axis=1)
        next_obs[np.arange(self.batch_size), seq_lengths - 1] = next_ob

        _, diag = tf_run(
            [self.train_op, self.train_diag],
            feed_dict={
                self.obs_ph: obs,
                self.act_ph: action,
                self.rew_ph: reward,
                self.next_obs_ph: next_obs,
                self.done_mask_ph: terminal
            }
        )

        tf_board_write(diag)

    def update_target(self) -> None:
        """ updates the target network """
        tf_run(self.update_target_op)

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
            return str(rnn_state[0][0][0])

        return str(rnn_state)
