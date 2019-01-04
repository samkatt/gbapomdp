""" ensemble of dqns """

import numpy as np
import tensorflow as tf

import agents.agent as agents
import networks.replay_buffer as rb
import networks.architectures as archs
import networks.DQNNet as DQNNet

from utils import tf_wrapper
from utils import misc

class ensemble_DQN(agents.Agent):
    """ DQN implementation"""

    # the policy from which to act e-greedy on right now
    _current_policy = 0

    def __init__(self, env, conf, sess, name='ensemble-dqn-agent'):
        """ initialize network """

        if conf.num_nets == 1:
            raise ValueError("no number of networks specified (--num_nets)")

        # confs
        self.target_update_freq = conf.q_target_update_freq
        self.batch_size = conf.batch_size
        self.train_freq = conf.train_frequency

        self.t = 0
        self.session = sess
        self.action_space = env.spaces()["A"]

        # construct the replay buffer
        self.replay_buffer = rb.ReplayBuffer(
            conf.replay_buffer_size,
            conf.observation_len, True)

        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
        arch = archs.TwoHiddenLayerQNet(conf)

        self.nets = []
        for i in range(conf.num_nets):
            self.nets.append(
                DQNNet.DQNNet(
                    env.spaces(),
                    arch,
                    optimizer,
                    conf,
                    sess,
                    name + '_net_' + str(i)))

    def reset(self, obs):
        """ resets to finish episode """
        self.replay_index = self.replay_buffer.store_frame(obs)
        self.latest_obs = self.replay_buffer.encode_recent_observation()

        self._current_policy = np.random.randint(0, len(self.nets)-1)

    def select_action(self):
        """ requests greedy action from network """

        # [fixme] too naive:
        # current q values are just randomly picked from the network
        q_values = self.nets[self._current_policy].Qvalues(self.latest_obs)

        self.latest_action = q_values.argmax()

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

        # batch update all networks
        if self.replay_buffer.can_sample(self.batch_size) and self.t % self.train_freq == 0:
            for net in self.nets:

                obs, actions, rewards, next_obs, done_mask = \
                        self.replay_buffer.sample(self.batch_size)

                net.batch_update(obs, actions, rewards, next_obs, done_mask)

        # update all target networks occasionally
        if self.t % self.target_update_freq == 0:
            for net in self.nets:
                net.update_target()

        self.t = self.t + 1
