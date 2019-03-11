""" agents """

import abc
import numpy as np
import tensorflow as tf

from agents.networks.neural_network_misc import ReplayBuffer
from misc import PiecewiseSchedule, epsilon_greedy


class Agent(abc.ABC):
    """ all agents must implement this interface """

    @abc.abstractmethod
    def reset(self, obs):
        """ called after each episode to prepare for the next

        Args:
             obs: the observation of the start of the episode

        """

    @abc.abstractmethod
    def select_action(self):
        """ asks the agent to select an action """

    @abc.abstractmethod
    def update(self, _observation, _reward: float, _terminal: bool):
        """ calls at the end of a real step to allow the agent to update

        The provided observation, reward and terminal are the result of a step
        in the real world given the last action

        Args:
             _observation: the previous observation of the step
             _reward: (`float`): the reward associated with the last step
             _terminal: (`bool`): whether the last step was terminal

        """


class EnsembleAgent(Agent):
    """ ensemble agent """

    t = 0
    last_ob = 0
    _storing_rbs = 0
    latest_action = 0

    # the policy from which to act e-greedy on right now
    _current_policy = 0

    # FIXME: take specific arguments instead of conf
    def __init__(self,
                 qnet_constructor,
                 arch,
                 env,
                 conf,
                 name='ensemble-agent'):
        """ initialize network """

        if conf.num_nets == 1:
            raise ValueError("no number of networks specified (--num_nets)")

        # consts
        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)

        # confs
        self.target_update_freq = conf.q_target_update_freq
        self.batch_size = conf.batch_size
        self.train_freq = conf.train_frequency

        # construct the replay buffer
        self.replay_buffers = np.array([
            {'index': 0, 'buffer': ReplayBuffer(
                conf.replay_buffer_size,
                conf.observation_len, True)}
            for _ in range(conf.num_nets)
        ])

        self.nets = [
            qnet_constructor(
                env.spaces(),
                arch,
                optimizer,
                conf,
                name + '_net_' + str(i))
            for i in range(conf.num_nets)
        ]

    def reset(self, obs):
        """ prepares for next episode

        stores the observation and resets its Q-network and resets its models

        Args:
             obs: the observation at the start of the episode

        """

        self.last_ob = obs

        for net in self.nets:
            net.reset()

        self._current_policy = np.random.randint(0, len(self.nets) - 1)

        # update which buffers are storing this episode
        self._storing_rbs = np.random.rand(len(self.replay_buffers)) > .5
        # make sure current is tracking
        self._storing_rbs[self._current_policy] = True

        for rb in self.replay_buffers[self._storing_rbs]:
            rb['index'] = rb['buffer'].store_frame(self.last_ob)

    def select_action(self):
        """ returns greedy action from current active policy """

        q_in = np.array([self.last_ob]) if self.nets[self._current_policy].is_recurrent(
        ) else self.replay_buffers[self._current_policy]['buffer'].encode_recent_observation()

        q_values = self.nets[self._current_policy].qvalues(q_in)

        self.latest_action = q_values.argmax()

        return self.latest_action

    def update(self, obs, reward: float, terminal: bool):
        """ stores the transition and potentially updates models

        For each network, does

        * Stores the experience (together with stored action) into the buffer with some probability
        * May perform a batch update (every so often, see parameters/configuration)
        * May update the target network (every so often, see parameters/configuration)

        Args:
             obs: the observation from the last step
             reward: (`float`): the reward of the last step
             terminal: (`bool`): whether the last step was terminal

        """

        # store experience
        for rb in self.replay_buffers[self._storing_rbs]:
            rb['buffer'].store_effect(
                rb['index'],
                self.latest_action,
                reward,
                terminal
            )

        self.last_ob = obs

        # store experience
        for rb in self.replay_buffers[self._storing_rbs]:
            rb['index'] = rb['buffer'].store_frame(self.last_ob)

        # batch update all networks using there respective replay buffer
        for net, rb in zip(self.nets, self.replay_buffers):
            if rb['buffer'].can_sample(
                    self.batch_size) and self.t % self.train_freq == 0:

                obs, actions, rewards, next_obs, done_mask = \
                    rb['buffer'].sample(self.batch_size)

                net.batch_update(obs, actions, rewards, next_obs, done_mask)

        # update all target networks occasionally
        if self.t % self.target_update_freq == 0:
            for net in self.nets:
                net.update_target()

        self.t = self.t + 1


class BaselineAgent(Agent):
    """ default single q-net agent implementation"""

    last_ob = 0
    replay_index = 0
    latest_action = 0

    def __init__(self,
                 qnet_constructor,
                 arch,
                 env,
                 conf,
                 exploration=None,
                 name='baseline-agent'):
        """ initialize network

        FIXME: take specific arguments instead of conf

        """

        # determines the e-greedy parameter over time
        self.exploration = exploration if exploration is not None else \
            PiecewiseSchedule([(0, 1.0), (2e4, 0.1), (1e5, 0.05)], outside_value=0.05)

        # how often the target network is being updated (every nth step)
        self.target_update_freq = conf.q_target_update_freq

        # the size of a batch update
        self.batch_size = conf.batch_size

        # how often the agent trains its network (every nth step)
        self.train_freq = conf.train_frequency

        # number of steps agent has taken
        self.t = 0

        # the entire action space, used to randomly pick actions
        self.action_space = env.spaces()["A"]

        # stores history of the agent
        self.replay_buffer = ReplayBuffer(
            conf.replay_buffer_size, conf.observation_len, True)

        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)

        # the q-function of the agent (estimation of the value of each action)
        self.q_net = qnet_constructor(
            env.spaces(),
            arch,
            optimizer,
            conf,
            name + '_net')

    def reset(self, obs):
        """ prepares for next episode

        stores the observation and resets its Q-network

        Args:
             obs: the observation at the start of the episode

        """
        self.last_ob = obs
        self.replay_index = self.replay_buffer.store_frame(self.last_ob)

        self.q_net.reset()

    def select_action(self):
        """ requests greedy action from network """

        if self.q_net.is_recurrent():
            q_in = np.array([self.last_ob])
        else:
            q_in = self.replay_buffer.encode_recent_observation()

        q_values = self.q_net.qvalues(q_in)

        epsilon = self.exploration.value(self.t)
        self.latest_action = epsilon_greedy(
            q_values, epsilon, self.action_space)

        return self.latest_action

    def update(self, obs, reward: float, terminal: bool):
        """ stores the transition and potentially updates models

        Stores the experience (together with stored action) into the buffer with some probability

        May perform a batch update (every so often, see parameters/configuration)

        May update the target network (every so often, see parameters/configuration)

        Args:
             obs: the observation from the last step
             reward: (`float`): the reward of the last step
             terminal: (`bool`): whether the last step was terminal

        """

        # store experience
        self.replay_buffer.store_effect(
            self.replay_index,
            self.latest_action,
            reward,
            terminal)

        self.last_ob = obs
        self.replay_index = self.replay_buffer.store_frame(self.last_ob)

        # batch update
        if self.replay_buffer.can_sample(
                self.batch_size) and self.t % self.train_freq == 0:
            obs, actions, rewards, next_obs, done_mask = self.replay_buffer.sample(
                self.batch_size)
            self.q_net.batch_update(obs, actions, rewards, next_obs, done_mask)

        # update target network occasionally
        if self.t % self.target_update_freq == 0:
            self.q_net.update_target()

        self.t = self.t + 1
