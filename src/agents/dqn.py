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

    # [fixme] create separate functions for getting architectures and train methods
    def __init__(self, env, conf, sess, exploration=None, name='dqn-agent'):
        """ initialize network """

        self.exploration = exploration if exploration is not None else \
                misc.PiecewiseSchedule([(0, 1.0), (2e4, 0.1), (1e5, 0.05)], outside_value=0.05)

        # confs
        self.target_update_freq = conf.q_target_update_freq
        self.batch_size = conf.batch_size
        self.train_freq = conf.train_frequency

        # init
        self.t = 0
        self.session = sess
        self.action_space = env.spaces()["A"]

        # construct the replay buffer
        self.replay_buffer = rb.ReplayBuffer(
            conf.replay_buffer_size,
            conf.observation_len, True)

        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
        q_net = archs.TwoHiddenLayerQNet(conf)

        # build q network
        input_shape = env.spaces()["O"].shape if conf.observation_len == 1 \
                else (conf.observation_len, *env.spaces()["O"].shape)

        self.obs_t_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        self.obs_tp1_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        self.qvalues = q_net(
            self.obs_t_ph,
            env.spaces()["A"].n,
            scope=name+'q_net')

        target_qvalues = q_net(
            self.obs_tp1_ph,
            env.spaces()["A"].n,
            scope=name+'target_q_net')

        # build train operation
        action_indices = tf.stack([tf.range(tf.size(self.act_t_ph)), self.act_t_ph], axis=-1)
        q_values = tf.gather_nd(self.qvalues, action_indices)
        targets = tf.reduce_max(target_qvalues, axis=-1)

        targets = tf.where(
            tf.cast(self.done_mask_ph, tf.bool),
            x=self.rew_t_ph, y=self.rew_t_ph + (conf.gamma * targets))

        # loss
        if conf.loss == "huber":
            loss = tf.losses.huber_loss(targets, q_values, delta=10.0)
        else:
            loss = tf.losses.mean_squared_error(targets, q_values)

        q_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name+'q_net')
        gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=q_net_vars))
        # clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 5)


        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # build target update operation
        update_target_fn = []
        target_q_net_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=name+'target_q_net')

        for var, var_target in zip(sorted(q_net_vars, key=lambda v: v.name),
                                   sorted(target_q_net_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))

        self.update_target_fn = tf.group(*update_target_fn)

        # finalize networks
        self.session.run(tf.global_variables_initializer())

    def reset(self, obs):
        """ resets to finish episode """
        self.replay_index = self.replay_buffer.store_frame(obs)
        self.latest_obs = self.replay_buffer.encode_recent_observation()

    def select_action(self):
        """ requests greedy action from network """
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
        if self.replay_buffer.can_sample(self.batch_size) and self.t % self.train_freq == 0:
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
