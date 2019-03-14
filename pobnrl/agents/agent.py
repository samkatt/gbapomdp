""" agents """

import abc
import numpy as np
import tensorflow as tf

from agents.networks.neural_network_misc import ReplayBuffer, Architecture
from agents.networks.q_functions import QNetInterface
from misc import epsilon_greedy, DiscreteSpace
from environments.environment import Environment


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
        """ asks the agent to select an action

        RETURNS: action

        """

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


class RandomAgent(Agent):
    """ Acts randomly """

    def __init__(self, action_space: DiscreteSpace):
        """ constructs an agent that will act randomly

        Args:
             num_actions: (`pobnrl.misc.DiscreteSpace`): the action space

        """
        self._action_space = action_space

    def reset(self, obs):
        """ Will not do anything since there is no internal state to reset

        Part of the interface of `pobnrl.agents.agent.Agent`

        Args:
             obs: ignored

        """

    def select_action(self):
        """ returns a random action

        Part of the interface of `pobnrl.agents.agent.Agent`

        RETURNS: the random action

        """
        return self._action_space.sample()

    def update(self, _observation, _reward: float, _terminal: bool):
        """ will not do anything since this has nothing to update

        Part of the interface of `pobnrl.agents.agent.Agent`

        Args:
             _observation: ignored
             _reward: (`float`): ignored
             _terminal: (`bool`): ignored

        """


# pylint: disable=too-many-instance-attributes
class BaselineAgent(Agent):
    """ default single q-net agent implementation"""

    replay_index = 0
    latest_action = 0

    def __init__(self,
                 qnet_constructor: QNetInterface,
                 arch: Architecture,
                 env: Environment,
                 **conf):
        """ initializes network

        Assumes conf contains:
            * `float` exploration (epsilon of e-greedy)
            * `int` q_target_update_freq
            * `int` batch_size
            * `int` train_frequency
            * `int` replay_buffer_size
            * `int` observation len
            * `float` learning rate (alpha)

        Args:
             qnet_constructor: (`QNetInterface`): Q-net to use
             arch: (`Architecture`): architecture to use
             env: (`Environment`): for domain knowledge
             **conf:

        """

        # determines the e-greedy parameter over time
        self.exploration = conf['exploration']

        # how often the target network is being updated (every nth step)
        self.target_update_freq = conf['q_target_update_freq']

        # the size of a batch update
        self.batch_size = conf['batch_size']

        # how often the agent trains its network (every nth step)
        self.train_freq = conf['train_frequency']

        # number of steps agent has taken
        self.t = 0  # pylint: disable=invalid-name

        # the entire action space, used to randomly pick actions
        self.action_space = env.spaces()["A"]

        # stores history of the agent
        self.replay_buffer = ReplayBuffer(
            conf['replay_buffer_size'], conf['observation_len'], True)

        optimizer = tf.train.AdamOptimizer(learning_rate=conf['learning_rate'])

        # the q-function of the agent (estimation of the value of each action)
        self.q_net = qnet_constructor(
            env.spaces(),
            arch,
            optimizer,
            **conf,
            scope=conf['name'] + '_net')

    def reset(self, obs):
        """ prepares for next episode

        stores the observation and resets its Q-network

        Args:
             obs: the observation at the start of the episode

        """
        self.replay_index = self.replay_buffer.store_frame(obs)
        self.q_net.reset()

    def select_action(self):
        """ requests greedy action from network """

        q_values = self.q_net.qvalues(
            self.replay_buffer.encode_recent_observation()
        )

        epsilon = self.exploration.value(self.t)
        self.latest_action = epsilon_greedy(
            q_values, epsilon, self.action_space)

        return self.latest_action

    def update(self, obs, reward: float, terminal: bool):
        """ stores the transition and potentially updates models

        Stores the experience into the buffers with some probability

        May perform a batch update (frequency: see parameters/configuration)

        May update the target network (frequency: see parameters/configuration)

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

        if not terminal:  # do not care about observations when episode ends
            self.replay_index = self.replay_buffer.store_frame(obs)

        # batch update
        if self.replay_buffer.can_sample(
                self.batch_size) and self.t % self.train_freq == 0:
            obs, actions, rewards, next_obs, done_mask =\
                self.replay_buffer.sample(self.batch_size)
            self.q_net.batch_update(obs, actions, rewards, next_obs, done_mask)

        # update target network occasionally
        if self.t % self.target_update_freq == 0:
            self.q_net.update_target()

        self.t = self.t + 1


class EnsembleAgent(Agent):
    """ ensemble agent """

    t = 0
    _storing_rbs = 0
    latest_action = 0

    # the policy from which to act e-greedy on right now
    _current_policy = 0

    def __init__(self,
                 qnet_constructor: QNetInterface,
                 arch: Architecture,
                 env: Environment,
                 **conf):
        """ initialize network

        Assumes conf contains:
            * `int` num_nets (> 1)
            * `float` exploration (epsilon of e-greedy)
            * `int` q_target_update_freq
            * `int` batch_size
            * `int` train_frequency
            * `int` replay_buffer_size
            * `int` observation len
            * `float` learning rate (alpha)

        Args:
             qnet_constructor: (`QNetInterface`): Q-net to use
             arch: (`Architecture`): architecture to use
             env: (`Environment`): for domain knowledge
             **conf:

        """

        assert conf['num_nets'] > 1, \
            "no number of networks specified (--num_nets)"

        # consts
        optimizer = tf.train.AdamOptimizer(learning_rate=conf['learning_rate'])

        # confs
        self.target_update_freq = conf['q_target_update_freq']
        self.batch_size = conf['batch_size']
        self.train_freq = conf['train_frequency']

        # construct the replay buffer
        self.replay_buffers = np.array([
            {'index': 0, 'buffer': ReplayBuffer(
                conf['replay_buffer_size'],
                conf['observation_len'], True)}
            for _ in range(conf['num_nets'])
        ])

        self.nets = [
            qnet_constructor(
                env.spaces(),
                arch,
                optimizer,
                **conf,
                scope=conf['name'] + '_net_' + str(i))
            for i in range(conf['num_nets'])
        ]

    def reset(self, obs):
        """ prepares for next episode

        stores the observation and resets its Q-network and resets its models

        Args:
             obs: the observation at the start of the episode

        """

        for net in self.nets:
            net.reset()

        self._current_policy = np.random.randint(0, len(self.nets) - 1)

        # update which buffers are storing this episode
        self._storing_rbs = np.random.rand(len(self.replay_buffers)) > .5
        # make sure current is tracking
        self._storing_rbs[self._current_policy] = True

        for replay_buffer in self.replay_buffers[self._storing_rbs]:
            replay_buffer['index'] = replay_buffer['buffer'].store_frame(obs)

    def select_action(self):
        """ returns greedy action from current active policy """

        q_values = self.nets[self._current_policy].qvalues(
            self.replay_buffers[self._current_policy]['buffer']
            .encode_recent_observation()
        )

        self.latest_action = q_values.argmax()

        return self.latest_action

    def update(self, obs, reward: float, terminal: bool):
        """ stores the transition and potentially updates models

        For each network, does

        * Stores the experience  into the buffer with some probability
        * May perform a batch update (frequency: see parameters/configuration)
        * May update target network (frequency see parameters/configuration)

        Args:
             obs: the observation from the last step
             reward: (`float`): the reward of the last step
             terminal: (`bool`): whether the last step was terminal

        """

        # store experience
        for replay_buffer in self.replay_buffers[self._storing_rbs]:
            replay_buffer['buffer'].store_effect(
                replay_buffer['index'],
                self.latest_action,
                reward,
                terminal
            )

        if not terminal:  # do not care about observations when episode ends
            for replay_buffer in self.replay_buffers[self._storing_rbs]:
                replay_buffer['index'] = replay_buffer['buffer'].store_frame(
                    obs
                )

        # batch update all networks using there respective replay buffer
        for net, rbuff in zip(self.nets, self.replay_buffers):
            if rbuff['buffer'].can_sample(
                    self.batch_size) and self.t % self.train_freq == 0:

                obs, actions, rewards, next_obs, done_mask = \
                    rbuff['buffer'].sample(self.batch_size)

                net.batch_update(obs, actions, rewards, next_obs, done_mask)

        # update all target networks occasionally
        if self.t % self.target_update_freq == 0:
            for net in self.nets:
                net.update_target()

        # pylint: disable=invalid-name
        self.t = self.t + 1
