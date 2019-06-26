""" Neural networks used as Q functions """

from typing import Callable, Optional, List
import abc
import numpy as np
import tensorflow as tf

from agents.neural_networks import misc, networks
from environments import ActionSpace
from misc import POBNRLogger, Space
from tf_api import tf_run, tf_board_write


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


class DQNNet(QNetInterface, POBNRLogger):
    """ a network based on DQN that can return q values and update """

    def __init__(
            self,
            action_space: ActionSpace,
            obs_space: Space,
            q_func: Callable[[np.ndarray, int, int], tf.Tensor],
            optimizer: tf.compat.v1.train.Optimizer,
            name: str,
            conf):
        """ construct the DRQNNet

        Assumes the input architecture q_func is **not** a recurrent one

        Args:
             action_space: (`pobnrl.environments.ActionSpace`): of environment
             obs_space: (`pobnrl.misc.Space`): of environment
             q_func: (`Callable`): the actual Q-function (non-recurrent)
             optimizer: the tf.optimizer to use for learning
             name: (`str`): name of the network (used for scoping)
             conf: (`namespace`): configurations

        """

        POBNRLogger.__init__(self)

        self.name = name
        self.history_len = conf.history_len
        self.batch_size = conf.batch_size

        self.replay_buffer = misc.ReplayBuffer()

        # shape of network input: variable in first dimension because
        # we sometimes provide complete sequences (for batch updates)
        # and sometimes just the last observation
        input_shape = (self.history_len, obs_space.ndim)

        # training operation place holders
        with tf.name_scope(self.name):
            self.act_ph = tf.compat.v1.placeholder(tf.int32, [None], name="actions")
            self.obs_ph = tf.compat.v1.placeholder(tf.float32, [None, *input_shape], name="obs")
            self.rew_ph = tf.compat.v1.placeholder(tf.float32, [None], name="rewards")
            self.done_mask_ph = tf.compat.v1.placeholder(tf.bool, [None], name="terminals")
            self.next_obs_ph = tf.compat.v1.placeholder(
                tf.float32, [None, *input_shape], name=f"{self.name}_next_obs"
            )

            # define operations to retrieve q and target values
            with tf.name_scope("net"):
                self.qvalues_fn = q_func(
                    self.obs_ph,
                    action_space.n,
                    conf.network_size,
                )

            with tf.name_scope("target"):
                next_targets_fn = q_func(
                    self.next_obs_ph,
                    action_space.n,
                    conf.network_size,
                )

            # define loss
            if conf.prior_function_scale != 0:
                assert conf.prior_function_scale > 0

                with tf.name_scope("prior_function"):
                    prior_vals = networks.simple_fc_nn(
                        self.obs_ph,
                        action_space.n,
                        4,
                    )

                    next_prior_vals = networks.simple_fc_nn(
                        self.next_obs_ph,
                        action_space.n,
                        4
                    )

                    scaled_prior = tf.scalar_mul(conf.prior_function_scale, prior_vals)
                    scaled_target_prior = tf.scalar_mul(
                        conf.prior_function_scale, next_prior_vals
                    )

                self.qvalues_fn = tf.add(self.qvalues_fn, scaled_prior)
                next_targets_fn = tf.add(next_targets_fn, scaled_target_prior)

            action_onehot = tf.stack(
                [tf.range(tf.size(self.act_ph)), self.act_ph], axis=-1, name='one_hot_actions'
            )

            q_values = tf.gather_nd(
                self.qvalues_fn,
                action_onehot,
                name="pick_Q"
            )

            targets = tf.where(
                self.done_mask_ph,
                x=self.rew_ph,
                y=self.rew_ph + (conf.gamma * tf.reduce_max(next_targets_fn, axis=-1))
            )

            loss = misc.loss(q_values, targets, conf.loss)

            net_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                scope=f'{tf.compat.v1.get_default_graph().get_name_scope()}/net'
            )

            gradients, variables = zip(
                *optimizer.compute_gradients(loss, var_list=net_vars)
            )

            if conf.clipping:
                gradients, _ = tf.clip_by_global_norm(gradients, 5)

            if conf.tensorboard_name:
                loss_summary = tf.compat.v1.summary.scalar('loss', tf.reduce_mean(loss))
                q_values_summary = tf.compat.v1.summary.histogram('q-values', q_values)

                grads = [tf.compat.v1.summary.scalar(grad.name, tf.sqrt(tf.reduce_mean(tf.square(grad))))
                         for grad in gradients]

                self.train_diag = tf.compat.v1.summary.merge([loss_summary, q_values_summary] + grads)
            else:
                self.train_diag = tf.no_op('no-diagnostics')

            self.train_op = optimizer.apply_gradients(zip(gradients, variables))

            # target update operation
            update_target_op = []
            target_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                scope=f"{tf.compat.v1.get_default_graph().get_name_scope()}/target"
            )

            for var, var_target in zip(sorted(net_vars, key=lambda v: v.name),
                                       sorted(target_vars, key=lambda v: v.name)):
                update_target_op.append(var_target.assign(var))

            self.update_target_op = tf.group(*update_target_op)

    def reset(self) -> None:
        """ resets the replay buffer

        Weights are controlled in tensorflow and reset outside of this scope

        """
        self.replay_buffer.clear()

    def episode_reset(self) -> None:
        """ no internal state so does nothing, interface requirement """

    def qvalues(self, obs: np.ndarray) -> np.array:
        """ returns the Q-values associated with the obs

        Pads the observation to the left if necessary to create correct shape
        Args:
             obs: (`np.ndarray`): the net input

        RETURNS (`np.array`): q-value for each action

        """

        assert obs.ndim >= 2, "observation expected to be len x shape"
        assert obs.shape[0] <= self.history_len

        if not len(obs) == self.history_len:
            padding = [(0, 0) for _ in range(len(obs[0].shape) + 1)]
            padding[0] = (self.history_len - len(obs), 0)

            obs = np.pad(obs, padding, 'constant')

        qvals = tf_run(self.qvalues_fn, feed_dict={self.obs_ph: obs[None]})

        self.log(POBNRLogger.LogLevel.V4, f"DQN: {obs} returned Q: {qvals}")

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

        # due to padding on the left side, we can simply append the *1*
        # next  observation to the original sequence and remove the first
        next_obs = np.concatenate((obs[:, 1:], np.zeros((self.batch_size, 1) + obs_shape)), axis=1)
        next_obs[:, -1] = next_ob

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

        assert conf.history_len > 0
        assert conf.batch_size > 0
        assert conf.network_size > 0
        assert 1 >= conf.gamma > 0
        assert conf.prior_function_scale >= 0

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

            self.rnn_state_ph \
                = tf.nn.rnn_cell.LSTMCell(conf.network_size).zero_state(
                    tf.shape(self.obs_ph)[0], dtype=tf.float32
                )

            # training operation q values and targets
            with tf.name_scope("net"):

                self.qvalues_fn, self.rec_state_fn = rec_q_func(
                    self.obs_ph,
                    self.rnn_state_ph,
                    action_space.n,
                    conf.network_size
                )

                action_onehot = tf.stack(
                    [tf.range(tf.size(self.act_ph)), self.act_ph], axis=-1
                )

                net_vars = tf.trainable_variables(scope=tf.get_default_graph().get_name_scope())

                if conf.prior_function_scale:
                    with tf.name_scope("prior_function"):

                        prior_vals, _ = networks.simple_fc_rnn(
                            self.obs_ph,
                            None,   # FIXME: should use rnn state too?!
                            action_space.n,
                            4,
                        )

                        scaled_prior = tf.scalar_mul(conf.prior_function_scale, prior_vals)
                        self.qvalues_fn = tf.add(self.qvalues_fn, scaled_prior)

                q_values = tf.gather_nd(self.qvalues_fn, action_onehot, name="pick_Q")

            # target network
            with tf.name_scope("target"):

                self.next_obs_ph = tf.compat.v1.placeholder(
                    tf.float32, [None, *input_shape], name="next_obs"
                )

                next_targets_fn, _ = rec_q_func(
                    self.next_obs_ph,
                    self.rnn_state_ph,  # FIXME: sometimes rnn_state? should not?
                    action_space.n,
                    conf.network_size
                )

                target_vars = tf.trainable_variables(scope=tf.get_default_graph().get_name_scope())

                if conf.prior_function_scale:
                    with tf.name_scope("prior_function"):

                        next_prior_vals, _ = networks.simple_fc_rnn(
                            self.next_obs_ph,
                            None,
                            action_space.n,
                            4
                        )

                        scaled_target_prior = tf.scalar_mul(conf.prior_function_scale, next_prior_vals)
                        next_targets_fn = tf.add(next_targets_fn, scaled_target_prior)

            with tf.name_scope('compute_target'):
                self.rew_ph = tf.compat.v1.placeholder(tf.float32, [None], name="rewards")
                self.done_mask_ph = tf.compat.v1.placeholder(tf.bool, [None], name="terminals")

                targets = tf.where(
                    self.done_mask_ph,
                    x=self.rew_ph,
                    y=self.rew_ph + (conf.gamma * tf.reduce_max(next_targets_fn, axis=-1))
                )

            loss = misc.loss(q_values, targets, conf.loss)

            gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=net_vars))

            if conf.clipping:  # FIXME: always do clipping, just make the default large
                gradients, _ = tf.clip_by_global_norm(gradients, 5)

            # FIXME: gotta do better than this check
            if conf.tensorboard_name:
                loss_summary = tf.compat.v1.summary.scalar('loss', tf.reduce_mean(loss))
                q_values_summary = tf.compat.v1.summary.histogram('q-values', q_values)

                grads = [
                    tf.compat.v1.summary.scalar(grad.name, tf.sqrt(tf.reduce_mean(tf.square(grad))))
                    for grad in gradients
                ]

                self.train_diag = tf.compat.v1.summary.merge([loss_summary, q_values_summary, grads])

            else:
                self.train_diag = tf.no_op('no-diagnostics')

            self.train_op = optimizer.apply_gradients(zip(gradients, variables))

            with tf.name_scope('update_target'):
                update_target_op = []
                for var, var_target in zip(sorted(net_vars, key=lambda v: v.name),
                                           sorted(target_vars, key=lambda v: v.name)):
                    update_target_op.append(var_target.assign(var))

                self.update_target_op = tf.group(*update_target_op)

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
            feed_dict[self.rnn_state_ph] = self.rnn_state

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
