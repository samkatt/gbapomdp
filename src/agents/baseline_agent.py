""" baseline (single Q network) implementation """

import numpy as np
import tensorflow as tf

import agents.agent as agents
import agents.networks.replay_buffer as rb
import agents.networks.architectures as archs
import agents.networks.qnets.DQNNet as DQNNet
import agents.networks.qnets.DRQNNet as DRQNNet

from utils import tf_wrapper
from utils import misc

class BaselineAgent(agents.Agent):
    """ default single q-net agent implementation"""

    def __init__(self,
                 qnet_constructor,
                 arch,
                 env,
                 conf,
                 exploration=None,
                 name='baseline-agent'):
        """ initialize network """

        self.exploration = exploration if exploration is not None else \
                misc.PiecewiseSchedule([(0, 1.0), (2e4, 0.1), (1e5, 0.05)], outside_value=0.05)

        # confs
        self.target_update_freq = conf.q_target_update_freq
        self.batch_size = conf.batch_size
        self.train_freq = conf.train_frequency

        # inits
        self.t = 0
        self.action_space = env.spaces()["A"]

        self.replay_buffer = rb.ReplayBuffer(
            conf.replay_buffer_size,
            conf.observation_len, True)

        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)

        self.q_net = qnet_constructor(
            env.spaces(),
            arch,
            optimizer,
            conf,
            name + '_net')

    def reset(self, obs):
        """ resets to finish episode """
        self.last_ob = obs
        self.replay_index = self.replay_buffer.store_frame(self.last_ob)

        self.q_net.reset()

    def select_action(self):
        """ requests greedy action from network """

        q_in = np.array([self.last_ob]) if self.q_net.is_recurrent() \
                else self.replay_buffer.encode_recent_observation()
        q_values = self.q_net.qvalues(q_in)

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

        self.last_ob = obs
        self.replay_index = self.replay_buffer.store_frame(self.last_ob)

        # batch update
        if self.replay_buffer.can_sample(self.batch_size) and self.t % self.train_freq == 0:
            obs, actions, rewards, next_obs, done_mask = self.replay_buffer.sample(self.batch_size)
            self.q_net.batch_update(obs, actions, rewards, next_obs, done_mask)

        # update target network occasionally
        if self.t % self.target_update_freq == 0:
            self.q_net.update_target()

        self.t = self.t + 1
