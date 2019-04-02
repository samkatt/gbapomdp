""" Neural networks used as Q functions

TODO: rename file
TODO: rethink creation of these classes in terms of arguments and names

"""

from typing import Callable
import abc
import logging
import numpy as np
import tensorflow as tf

from agents.networks import neural_network_misc
from misc import tf_run


class QNetInterface(abc.ABC):
    """ interface to all Q networks """

    # size of network
    SIZES = {"small": 16, "med": 64, "large": 512}

    @abc.abstractmethod
    def reset(self):
        """ resets to initial state """

    @abc.abstractmethod
    def episode_reset(self):
        """ resets the internal state to prepare for a new episode """

    @abc.abstractmethod
    def qvalues(self, obs: np.ndarray) -> np.array:
        """ returns the Q-values associated with the obs

        Args:
             obs: (`np.ndarray`): the net input

        RETURNS (`np.array`): q-value for each action

        """

    @abc.abstractmethod
    def batch_update(self):
        """ performs a batch update """

    @abc.abstractmethod
    def update_target(self):
        """ updates the target network """

    @abc.abstractmethod
    def record_transition(
            self,
            observation: np.ndarray,
            action: int,
            reward: float,
            terminal: bool):
        """ notifies this of provided transition

        Args:
             observation: (`np.ndarray`): shape depends on env
             action: (`int`): the taken action
             reward: (`float`): the resulting reward
             terminal: (`bool`): whether transition was terminal
        """


class DQNNet(QNetInterface):  # pylint: disable=too-many-instance-attributes
    """ a network based on DQN that can return q values and update """

    logger = logging.getLogger(__name__)

    def __init__(  # pylint: disable=too-many-locals
            self,
            env_spaces: dict,
            q_func: Callable,  # Q-value network function
            optimizer,
            **conf):
        """ construct the DRQNNet

        Assumes the input architecture q_func is **not** a recurrent one

        Args:
             env_spaces: (`dict`): {'O','A'} with observation and action space
             q_func: (`Callable`): the actual Q-function (non-recurrent)
             optimizer: the tf.optimizer to use for learning
             conf: configurations (assumptions above)

        Assumes conf contains:
            * `str` scope
            * `str` loss description
            * `str` network_size in {'small', 'med', 'large'}
            * `int` batch_size: size of batch learning
            * `int` history_len: length of history to consider
            * `bool` prior_functions
            * `bool` double_q
            * `bool` clipping
            * `float` gamma


        """

        assert conf['history_len'] > 0
        assert conf['batch_size'] > 0
        assert conf['network_size'] in self.SIZES.keys()
        assert 1 >= conf['gamma'] > 0

        self.name = conf['scope']
        self.history_len = conf['history_len']
        self.batch_size = conf['batch_size']

        self.replay_buffer \
            = neural_network_misc.ReplayBuffer(env_spaces["O"].shape)

        # shape of network input: variable in first dimension because
        # we sometimes provide complete sequences (for batch updates)
        # and sometimes just the last observation
        input_shape = (self.history_len, *env_spaces["O"].shape)

        # training operation place holders
        # TODO: rename these
        self.obs_t_ph = tf.placeholder(
            tf.float32, [None] + list(input_shape), name=self.name + '_obs'
        )
        self.act_t_ph = tf.placeholder(
            tf.int32, [None], name=self.name + '_actions'
        )
        self.rew_t_ph = tf.placeholder(
            tf.float32, [None], name=self.name + '_rewards'
        )
        self.done_mask_ph = tf.placeholder(
            tf.bool, [None], name=self.name + '_terminals'
        )
        self.obs_tp1_ph = tf.placeholder(
            tf.float32,
            [None] + list(input_shape),
            name=self.name + '_next_obs'
        )

        # define operations to retrieve q and target values

        self.qvalues_fn = q_func(
            self.obs_t_ph,
            env_spaces["A"].n,
            self.SIZES[conf['network_size']],
            scope=self.name + '_net'
        )

        next_qvalues_fn = q_func(
            self.obs_tp1_ph,
            env_spaces["A"].n,
            self.SIZES[conf['network_size']],
            scope=self.name + '_net'
        )

        next_targets_fn = q_func(
            self.obs_tp1_ph,
            env_spaces["A"].n,
            self.SIZES[conf['network_size']],
            scope=self.name + '_target'
        )

        # define loss
        if conf['prior_functions']:  # add random function to our estimates
            prior_vals = neural_network_misc.two_layer_q_net(
                self.obs_t_ph,
                env_spaces["A"].n,
                4,
                scope=self.name + '_prior'
            )

            next_prior_vals = neural_network_misc.two_layer_q_net(
                self.obs_tp1_ph,
                env_spaces["A"].n,
                4,
                scope=self.name + '_prior'
            )

            qvalues_fn = tf.add(self.qvalues_fn, prior_vals)
            next_targets_fn = tf.add(next_targets_fn, next_prior_vals)

        else:  # no random prior to our loss function
            qvalues_fn = self.qvalues_fn

        action_onehot = tf.stack(
            [tf.range(tf.size(self.act_t_ph)), self.act_t_ph], axis=-1
        )

        q_values = tf.gather_nd(
            qvalues_fn,
            action_onehot,
            name=self.name + '_pick_Q'
        )

        return_estimate = neural_network_misc.return_estimate(
            next_qvalues_fn,
            next_targets_fn,
            conf['double_q']
        )

        targets = tf.where(
            self.done_mask_ph,
            x=self.rew_t_ph,
            y=self.rew_t_ph + (conf['gamma'] * return_estimate)
        )

        loss = neural_network_misc.loss(q_values, targets, conf['loss'])

        net_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '_net'
        )
        gradients, variables = zip(
            *optimizer.compute_gradients(loss, var_list=net_vars)
        )

        if conf['clipping']:
            gradients, _ = tf.clip_by_global_norm(gradients, 5)

        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # target update operation
        update_target_op = []
        target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '_target'
        )

        for var, var_target in zip(sorted(net_vars, key=lambda v: v.name),
                                   sorted(target_vars, key=lambda v: v.name)):
            update_target_op.append(var_target.assign(var))

        self.update_target_op = tf.group(*update_target_op)

    def reset(self):
        """ resets the replay buffer

        Weights are controlled in tensorflow and reset outside of this scope

        """
        self.replay_buffer.clear()

    def episode_reset(self):
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

        return tf_run(
            self.qvalues_fn,
            feed_dict={self.obs_t_ph: obs[None]}  # add batch dim
        )

    def batch_update(self):
        """ performs a batch update """

        if self.replay_buffer.size < self.batch_size:
            self.logger.debug(
                "Network %s cannot batch update due to small buf", self.name
            )
            return

        batch = self.replay_buffer.sample(
            self.batch_size, self.history_len, padding='left'
        )

        tf_run(
            self.train_op,
            feed_dict={
                self.obs_t_ph: batch['obs'],
                self.act_t_ph: batch['actions'][:, -1],
                self.rew_t_ph: batch['rewards'][:, -1],
                self.obs_tp1_ph: batch['next_obs'],
                self.done_mask_ph: batch['terminals'][:, -1]
            }
        )

    def update_target(self):
        """ updates the target network """
        tf_run(self.update_target_op)

    def record_transition(
            self,
            observation: np.ndarray,
            action: int,
            reward: float,
            terminal: bool):
        """ notifies this of provided transition

        Stores transition in replay buffer

        Args:
             observation: (`np.ndarray`): shape depends on env
             action: (`int`): the taken action
             reward: (`float`): the resulting reward
             terminal: (`bool`): whether transition was terminal
        """

        self.replay_buffer.store(observation, action, reward, terminal)


class DRQNNet(QNetInterface):  # pylint: disable=too-many-instance-attributes
    """ a network based on DRQN that can return q values and update """

    logger = logging.getLogger(__name__)

    def __init__(  # pylint: disable=too-many-locals
            self,
            env_spaces: dict,
            rec_q_func: Callable,
            optimizer,
            **conf):
        """ construct the DRQNNet

        Assumes the rec_q_func provided is a recurrent Q function

        Args:
             env_spaces: (`dict`): {'O','A'} with observation and action space
             rec_q_func: (`Callable`): the (recurrent) Q function
             optimizer: the tf.optimizer optimizer to use for learning
             conf: configurations

        Assumes conf contains:
            * `str` scope: the name / scope of this
            * `str` network_size: in {'small', 'med', 'large'}
            * `str` loss: description of how to compute the loss
            * `int` batch_size: size of batch learning
            * `int` history_len: length of history to consider
            * `bool` prior_functions: whether to use random prior in target
            * `bool` double_q: whether to apply the double_q method
            * `bool` clipping: whether to apply clipping
            * `float` gamma: discount factor in target

        """

        assert conf['history_len'] > 0
        assert conf['batch_size'] > 0
        assert conf['network_size'] in self.SIZES.keys()
        assert 1 >= conf['gamma'] > 0

        self.name = conf['scope']
        self.history_len = conf['history_len']
        self.batch_size = conf['batch_size']

        self.replay_buffer \
            = neural_network_misc.ReplayBuffer(env_spaces["O"].shape)
        self.rnn_state = None

        # shape of network input: variable in first dimension because
        # we sometimes provide complete sequences (for batch updates)
        # and sometimes just the last observation
        input_shape = (None, *env_spaces["O"].shape)

        # training operation place holders
        self.obs_t_ph = tf.placeholder(
            tf.float32, [None] + list(input_shape), name=self.name + '_obs'
        )
        self.act_t_ph = tf.placeholder(
            tf.int32, [None], name=self.name + '_actions'
        )
        self.rew_t_ph = tf.placeholder(
            tf.float32, [None], name=self.name + '_rewards'
        )
        self.done_mask_ph = tf.placeholder(
            tf.bool, [None], name=self.name + '_terminals'
        )
        self.obs_tp1_ph = tf.placeholder(
            tf.float32,
            [None] + list(input_shape),
            name=self.name + '_next_obs'
        )

        rnn_cell = tf.nn.rnn_cell.LSTMCell(self.SIZES[conf['network_size']])
        rnn_cell_t = tf.nn.rnn_cell.LSTMCell(self.SIZES[conf['network_size']])

        self.rnn_state_ph = rnn_cell.zero_state(
            tf.shape(self.obs_t_ph)[0], dtype=tf.float32
        )

        self.seq_lengths_ph = tf.placeholder(
            tf.int32, [None], name=self.name + '_seq_len'
        )

        # training operation q values and targets
        self.qvalues_fn, self.rec_state_fn = rec_q_func(
            self.obs_t_ph,
            self.seq_lengths_ph,
            rnn_cell,
            self.rnn_state_ph,
            env_spaces["A"].n,
            self.SIZES[conf['network_size']],
            scope=self.name + '_net'
        )

        next_targets_fn, _ = rec_q_func(
            self.obs_tp1_ph,
            self.seq_lengths_ph,
            rnn_cell_t,
            self.rnn_state_ph,
            env_spaces["A"].n,
            self.SIZES[conf['network_size']],
            scope=self.name + '_target'
        )

        next_qvalues_fn, _ = rec_q_func(
            self.obs_tp1_ph,
            self.seq_lengths_ph,
            rnn_cell,
            self.rnn_state_ph,
            env_spaces["A"].n,
            self.SIZES[conf['network_size']],
            scope=self.name + '_net'
        )

        # define loss

        if conf['prior_functions']:  # add random function to our estimates

            prior_vals, _ = neural_network_misc.two_layer_rec_q_net(
                self.obs_t_ph,
                self.seq_lengths_ph,
                rnn_cell,
                self.rnn_state_ph,
                env_spaces["A"].n,
                4,
                scope=self.name + '_prior'
            )
            next_prior_vals, _ = neural_network_misc.two_layer_rec_q_net(
                self.obs_tp1_ph,
                self.seq_lengths_ph,
                self.rnn_state_ph,
                env_spaces["A"].n,
                4,
                scope=self.name + '_prior'
            )

            qvalues_fn = tf.add(self.qvalues_fn, prior_vals)
            next_targets_fn = tf.add(next_targets_fn, next_prior_vals)

        else:  # no random prior to our loss function
            qvalues_fn = self.qvalues_fn

        action_onehot = tf.stack(
            [tf.range(tf.size(self.act_t_ph)), self.act_t_ph], axis=-1
        )

        q_values = tf.gather_nd(
            qvalues_fn,
            action_onehot,
            name=self.name + '_pick_Q'
        )

        return_estimate = neural_network_misc.return_estimate(
            next_qvalues_fn,
            next_targets_fn,
            conf['double_q']
        )

        targets = tf.where(
            self.done_mask_ph,
            x=self.rew_t_ph,
            y=self.rew_t_ph + (conf['gamma'] * return_estimate)
        )

        loss = neural_network_misc.loss(q_values, targets, conf['loss'])

        net_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '_net'
        )
        gradients, variables = zip(
            *optimizer.compute_gradients(loss, var_list=net_vars)
        )

        if conf['clipping']:
            gradients, _ = tf.clip_by_global_norm(gradients, 5)

        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # target update operation
        update_target_op = []
        target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '_target'
        )

        for var, var_target in zip(sorted(net_vars, key=lambda v: v.name),
                                   sorted(target_vars, key=lambda v: v.name)):
            update_target_op.append(var_target.assign(var))

        self.update_target_op = tf.group(*update_target_op)

    def reset(self):
        """ resets the net internal state and replay buffer

        Weights are controlled in tensorflow and reset outside of this scope

        """

        self.replay_buffer.clear()
        self.rnn_state = None

    def episode_reset(self):
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
            self.obs_t_ph: obs[-1, None, None],  # cast last ob to shape
            self.seq_lengths_ph: np.array([1])  # just a single step 'seq'
        }

        if self.rnn_state is not None:
            feed_dict[self.rnn_state_ph] = self.rnn_state

        qvals, self.rnn_state = tf_run(
            [self.qvalues_fn, self.rec_state_fn],
            feed_dict=feed_dict
        )

        return qvals

    def batch_update(self):
        """ performs a batch update """

        if self.replay_buffer.size < self.batch_size:
            self.logger.debug(
                "Network % cannot batch update due to small buf", self.name
            )
            return

        batch = self.replay_buffer.sample(
            self.batch_size, self.history_len, padding='right'
        )

        sampled_step_idx = \
            np.eye(self.history_len, dtype=bool)[batch['seq_lengths'] - 1]

        tf_run(
            self.train_op,
            feed_dict={
                self.seq_lengths_ph: batch['seq_lengths'],
                self.obs_t_ph: batch['obs'],
                self.act_t_ph: batch['actions'][sampled_step_idx],
                self.rew_t_ph: batch['rewards'][sampled_step_idx],
                self.obs_tp1_ph: batch['next_obs'],
                self.done_mask_ph: batch['terminals'][sampled_step_idx]
            }
        )

    def update_target(self):
        """ updates the target network """
        tf_run(self.update_target_op)

    def record_transition(
            self,
            observation: np.ndarray,
            action: int,
            reward: float,
            terminal: bool):
        """ notifies this of provided transition

        Stores transition in replay buffer

        Args:
             observation: (`np.ndarray`): shape depends on env
             action: (`int`): the taken action
             reward: (`float`): the resulting reward
             terminal: (`bool`): whether transition was terminal
        """
        self.replay_buffer.store(observation, action, reward, terminal)
