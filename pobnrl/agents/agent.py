""" agents

"""

from collections import deque
from typing import Callable
import abc
import numpy as np
import tensorflow as tf

from agents.networks.q_functions import QNetInterface
from environments.environment import Environment
from misc import PiecewiseSchedule, epsilon_greedy, DiscreteSpace


class Agent(abc.ABC):
    """ all agents must implement this interface """

    @abc.abstractmethod
    def reset(self):
        """ resets agent to initial state """

    @abc.abstractmethod
    def episode_reset(self, observation: np.ndarray):
        """ called after each episode to prepare for the next

        Args:
             observation: (`np.ndarray`): the initial episode observation

        """

    @abc.abstractmethod
    def select_action(self):
        """ asks the agent to select an action

        RETURNS: action

        """

    @abc.abstractmethod
    def update(
            self,
            observation: np.ndarray,
            reward: float,
            terminal: bool):
        """ calls at the end of a real step to allow the agent to update

        The provided observation, reward and terminal are the result of a step
        in the real world given the last action

        Args:
             observation (`np.ndarray`): the observation
             reward: (`float`): the reward associated with the last step
             terminal: (`bool`): whether the last step was terminal

        """


class RandomAgent(Agent):
    """ Acts randomly """

    def __init__(self, action_space: DiscreteSpace):
        """ constructs an agent that will act randomly

        Args:
             num_actions: (`pobnrl.misc.DiscreteSpace`): the action space

        """
        self._action_space = action_space

    def reset(self):
        """ stateless and thus ignored """

    def episode_reset(self, observation: np.ndarray):
        """ Will not do anything since there is no internal state to reset

        Part of the interface of `pobnrl.agents.agent.Agent`

        Args:
             observation: (`np.ndarray`): ignored

        """

    def select_action(self):
        """ returns a random action

        Part of the interface of `pobnrl.agents.agent.Agent`

        RETURNS: the random action

        """
        return self._action_space.sample()

    def update(
            self,
            observation: np.ndarray,
            reward: float,
            terminal: bool):
        """ will not do anything since this has nothing to update

        Part of the interface of `pobnrl.agents.agent.Agent`

        Args:
             observation (`np.ndarray`): ignored
             reward: (`float`): ignored
             terminal: (`bool`): ignored

        """


class BaselineAgent(Agent):  # pylint: disable=too-many-instance-attributes
    """ default single q-net agent implementation"""

    def __init__(self,
                 qnet_constructor: QNetInterface,
                 q_func: Callable,
                 env: Environment,
                 **conf):
        """ initializes a single network

        Args:
             qnet_constructor: (`QNetInterface`): Q-net to use
             q_func: (`Callable`): Q-function (network arch) to use
             env: (`Environment`): for domain knowledge
             **conf: set of configurations

        Assumes conf contains:
            * `int` target_update_freq
            * `int` train_freq
            * `int` observation len
            * `float` learning rate (alpha)
            * whatever is necessary for the `q_func`

        """

        assert conf['target_update_freq'] > 0
        assert conf['train_freq'] > 0
        assert 1 > conf['learning_rate'] > 0
        assert conf['history_len'] > 0

        # params
        self.target_update_freq = conf['target_update_freq']
        self.train_freq = conf['train_freq']

        self.action_space = env.action_space

        self.timestep = 0
        self.last_action = None
        self.last_obs = deque([], conf['history_len'])

        self.exploration = PiecewiseSchedule(
            [(0, 1.0), (2e4, 0.1), (1e5, 0.05)], outside_value=0.05
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=conf['learning_rate'])

        self.q_net = qnet_constructor(
            {
                "A": env.action_space,
                "O": env.observation_space
            },
            q_func,
            optimizer,
            **conf,
            scope=conf['name'] + '_net'
        )

    def reset(self):
        """ re-initializes members

        e.g.  empties replay buffer and sets timestep to 0, resets net
        """

        self.timestep = 0
        self.last_action = None
        self.last_obs.clear()
        self.q_net.reset()

    def episode_reset(self, observation: np.ndarray):
        """ prepares for next episode

        stores the observation and resets its Q-network

        Args:
             observation: (`np.ndarray`): stored and later given to QNet

        """

        self.last_obs.clear()
        self.last_obs.append(observation)

        self.q_net.episode_reset()

    def select_action(self):
        """ requests greedy action from network """

        q_values = self.q_net.qvalues(np.array(self.last_obs))

        epsilon = self.exploration.value(self.timestep)
        self.last_action = epsilon_greedy(q_values, epsilon, self.action_space)

        return self.last_action

    def update(
            self,
            observation: np.ndarray,
            reward: float,
            terminal: bool):
        """ stores the transition and potentially updates models

        Stores the experience into the buffers with some probability

        May perform a batch update (frequency: see parameters/configuration)

        May update the target network (frequency: see parameters/configuration)


        Args:
             observation (`np.ndarray`): the observation
             reward: (`float`): the reward associated with the last step
             terminal: (`bool`): whether the last step was terminal

        """

        self.q_net.record_transition(
            self.last_obs[-1], self.last_action, reward, terminal
        )

        self.last_obs.append(observation)

        if self.timestep % self.train_freq == 0:
            self.q_net.batch_update()

        if self.timestep % self.target_update_freq == 0:
            self.q_net.update_target()

        self.timestep += 1


class EnsembleAgent(Agent):  # pylint: disable=too-many-instance-attributes
    """ ensemble agent """

    def __init__(self,
                 qnet_constructor: QNetInterface,
                 q_func: Callable,
                 env: Environment,
                 **conf):
        """ initialize network

        Args:
             qnet_constructor: (`QNetInterface`): Q-net to use
             q_func: (`Callable`): Q-function (network arch) to use
             env: (`Environment`): for domain knowledge
             **conf: set of configurations

        Assumes conf contains:
            * `int` num_nets (> 1)
            * `int` target_update_freq
            * `int` train_freq
            * `int` observation len
            * `float` learning rate (alpha)

        """

        assert conf['num_nets'] > 1
        assert conf['target_update_freq'] > 0
        assert conf['train_freq'] > 0
        assert 1 > conf['learning_rate'] > 0
        assert conf['history_len'] > 0

        # params
        self.target_update_freq = conf['target_update_freq']
        self.train_freq = conf['train_freq']

        self.timestep = 0
        self.last_action = None
        self.last_obs = deque([], conf['history_len'])

        optimizer = tf.train.AdamOptimizer(learning_rate=conf['learning_rate'])

        self.nets = np.array([
            qnet_constructor(
                {
                    "A": env.action_space,
                    "O": env.observation_space
                },
                q_func,
                optimizer,
                **conf,
                scope=conf['name'] + '_net_' + str(i))
            for i in range(conf['num_nets'])
        ])

        self._storing_nets = self.nets[np.random.rand(len(self.nets)) > .5]
        self._current_policy = np.random.choice(self.nets)

    def reset(self):
        """ re-initializes members

        e.g.  empties replay buffer and sets timestep to 0, resets nets
        """

        self.timestep = 0
        self.last_action = None
        self.last_obs.clear()

        self._storing_nets = self.nets[np.random.rand(len(self.nets)) > .5]
        self._current_policy = np.random.choice(self.nets)

        for net in self.nets:
            net.reset()

    def episode_reset(self, observation: np.ndarray):
        """ prepares for next episode

        stores the observation and resets its Q-network and resets its models

        Args:
             observation: (`np.ndarray`):  stored and later given to QNet

        """

        self.last_obs.clear()
        self.last_obs.append(observation)

        self._storing_nets = self.nets[np.random.rand(len(self.nets)) > .5]
        self._current_policy = np.random.choice(self.nets)

        for net in self.nets:
            net.episode_reset()

    def select_action(self):
        """ returns greedy action from current active policy """

        self.last_action = self._current_policy.qvalues(
            np.array(self.last_obs)
        ).argmax()

        return self.last_action

    def update(
            self,
            observation: np.ndarray,
            reward: float,
            terminal: bool):
        """ stores the transition and potentially updates models

        For each network, does

        * Stores the experience  into the buffer with some probability
        * May perform a batch update (frequency: see parameters/configuration)
        * May update target network (frequency see parameters/configuration)

        Args:
             observation (`np.ndarray`): the observation
             reward: (`float`): the reward associated with the last step
             terminal: (`bool`): whether the last step was terminal

        """

        # store experience for recording nets
        for net in self._storing_nets:
            net.record_transition(
                self.last_obs[-1], self.last_action, reward, terminal
            )

        self.last_obs.append(observation)

        if self.timestep % self.train_freq == 0:
            for net in self.nets:
                net.batch_update()

        if self.timestep % self.target_update_freq == 0:
            for net in self.nets:
                net.update_target()

        self.timestep += 1
