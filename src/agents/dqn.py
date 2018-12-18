""" dqn implementation """

import numpy as np
import tensorflow as tf

import agents.agent as agents
import networks.replay_buffer as rb
import networks.architectures as archs

from utils import tf_wrapper
from utils import misc

class DQN(agents.Agent):
    """ DQN implementation"""

    def __init__(self, env, conf):
        """ initialize network """

        # self.exploration = misc.PiecewiseSchedule(
            # [(0, 1.0), (2e5, 0.1)],
            # outside_value=0.1)

        self.exploration = misc.PiecewiseSchedule(
            [(0, 1.0), (2e4, 0.1), (2e5, 0.05)],
            outside_value=0.05)

        # init
        self.t = 0
        self.session = tf_wrapper.get_session()
        self.action_space = env.spaces()["A"]

        # confs
        self.target_update_freq = conf.q_target_update_freq
        self.batch_size = conf.batch_size
        self.train_freq = conf.train_frequency
        self.explore_duration = conf.explore_duration

        # construct the replay buffer
        self.replay_buffer = rb.ReplayBuffer(
            conf.replay_buffer_size,
            conf.observation_len, True)

        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
        q_net = archs.TwoHiddenLayerQNet(conf)

        # build q network
        input_shape = (conf.observation_len, *env.spaces()["O"].shape)

        self.obs_t_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        self.obs_tp1_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        self.qvalues = q_net(
            self.obs_t_ph,
            env.spaces()["A"].n,
            scope='q_net')

        target_qvalues = q_net(
            self.obs_tp1_ph,
            env.spaces()["A"].n,
            scope='target_q_net')

        # build train operation
        action_indices = tf.stack([tf.range(tf.size(self.act_t_ph)), self.act_t_ph], axis=-1)
        onpolicy_qvalues = tf.gather_nd(self.qvalues, action_indices)
        targets = tf.reduce_max(target_qvalues, axis=-1)

        done_td_error = self.rew_t_ph - onpolicy_qvalues

        not_done_td_error = done_td_error + (conf.gamma * targets)

        td_error = tf.where(
            tf.cast(self.done_mask_ph, tf.bool),
            x=done_td_error, y=not_done_td_error)

        total_error = tf.reduce_mean(tf.square(td_error))

        q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_net')
        grads_and_vars = optimizer.compute_gradients(total_error, var_list=q_net_vars)

        self.train_op = optimizer.apply_gradients(grads_and_vars)

        # build target update operation
        update_target_fn = []
        target_q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_net')
        for var, var_target in zip(sorted(q_net_vars, key=lambda v: v.name),
                                   sorted(target_q_net_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

        # finalize networks
        self.session.run(tf.global_variables_initializer())

    def reset(self, obs):
        """ resets the rnn state """
        self.replay_index = self.replay_buffer.store_frame(obs)
        self.latest_obs = self.replay_buffer.encode_recent_observation()

    def select_action(self):
        """ requests greedy action from network """
        # [checkme] why None?
        q_values = self.session.run(self.qvalues, feed_dict={self.obs_t_ph: self.latest_obs[None]})
        epsilon = self.exploration.value(self.t)

        self.latest_action = misc.epsilon_greedy(q_values, epsilon, self.action_space)

        return self.latest_action

    def update(self, obs, reward, terminal):
        """ store experience and batch update """

        # store experience
        self.replay_buffer.store_effect(
            self.replay_index,
            self.latest_action,
            reward,
            terminal)

        self.replay_index = self.replay_buffer.store_frame(obs)
        self.latest_obs = self.replay_buffer.encode_recent_observation()

        # batch update
        if self.t > self.explore_duration and self.t % self.train_freq == 0:
            obs, actions, rewards, next_obs, done_mask = self.replay_buffer.sample(self.batch_size)

            self.session.run(self.train_op, feed_dict={
                self.obs_t_ph: obs,
                self.act_t_ph: actions,
                self.rew_t_ph: rewards,
                self.obs_tp1_ph: next_obs,
                self.done_mask_ph: done_mask})

        # update target network occasionally
        if self.t % self.target_update_freq == 0:
            self.session.run(self.update_target_fn)

        self.t = self.t + 1
