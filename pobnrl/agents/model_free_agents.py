""" agents that learn a Q function directly """

from collections import deque
from typing import Callable
import numpy as np

from misc import ActionSpace, epsilon_greedy, ExplorationSchedule
from misc import PiecewiseSchedule, FixedExploration, DiscreteSpace

from .agent import Agent
from .networks import create_qnet
from .networks.q_functions import QNetInterface


class BaselineAgent(Agent):  # pylint: disable=too-many-instance-attributes
    """ default single q-net agent implementation"""

    def __init__(self,
                 qnet: QNetInterface,
                 action_space: ActionSpace,
                 exploration: ExplorationSchedule,
                 conf):
        """ initializes a single network

        Args:
             qnet: (`pobnrl.agents.networks.q_functions.QNetInterface`): \
                    Q-net to use
             action_space: (`pobnrl.misc.ActionSpace`): of environment
             exploration: (`pobnrl.misc.ExplorationSchedule`): \
                    schedule for e-greedy
             conf: (`namespace`) set of configurations (see -h)

        """

        assert conf.target_update_freq > 0
        assert conf.train_freq > 0
        assert conf.history_len > 0

        # params
        self.target_update_freq = conf.target_update_freq
        self.train_freq = conf.train_freq

        self.action_space = action_space

        self.timestep = 0
        self.last_action = None
        self.last_obs = deque([], conf.history_len)

        self.exploration = exploration

        self.q_net = qnet

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

    def select_action(self) -> int:
        """ requests greedy action from network

        RETURNS (`int`):

        """

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
                 qnet_constructor: Callable[[str], QNetInterface],
                 action_space: ActionSpace,
                 exploration: ExplorationSchedule,
                 conf):
        """ initialize network

        Args:
            qnet_constructor: (`Callable`[[`str`], `pobnrl.agents.networks.q_functions.QNetInterface`]): \
                    Q-net constructor to use to create nets (given scope)
            action_space: (`pobnrl.misc.ActionSpace`): of environment
            exploration: (`pobnrl.misc.ExplorationSchedule`): \
                    exploration schedule
            conf: (`namespace`): set of configurations

        """

        assert conf.num_nets > 1
        assert conf.target_update_freq > 0
        assert conf.train_freq > 0
        assert conf.history_len > 0

        # params
        self.target_update_freq = conf.target_update_freq
        self.train_freq = conf.train_freq

        self.action_space = action_space
        self.exploration = exploration

        self.timestep = 0
        self.last_action = None
        self.last_obs = deque([], conf.history_len)

        self.nets = np.array(
            [qnet_constructor('ensemble_net_' + str(i))
             for i in range(conf.num_nets)]
        )

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

    def select_action(self) -> int:
        """ returns greedy action from current active policy

        RETURNS (`int`):

        """

        q_values = self._current_policy.qvalues(np.array(self.last_obs))

        epsilon = self.exploration.value(self.timestep)
        self.last_action = epsilon_greedy(q_values, epsilon, self.action_space)

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
             observation: (`np.ndarray`): the observation
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


def create_agent(
        action_space: DiscreteSpace,
        observation_space: DiscreteSpace,
        conf) -> Agent:
    """ factory function to construct model-free learning agents

    Args:
         action_space: (`pobnrl.misc.DiscreteSpace`): of environment
         observation_space: (`pobnrl.misc.DiscreteSpace`) of environment
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.agent.Agent`)

    """

    if 1 >= conf.exploration >= 0:
        exploration_schedule = FixedExploration(conf.exploration)
    else:
        exploration_schedule = PiecewiseSchedule(
            [(0, 1.0), (2e4, 0.1), (1e5, 0.05)], outside_value=0.05
        )

    if conf.num_nets == 1:
        # single-net agent

        return BaselineAgent(
            create_qnet(
                action_space,
                observation_space,
                'q_net',
                conf
            ),
            action_space,
            exploration_schedule,
            conf
        )

    # num_nets > 1: ensemble agent

    def qnet_constructor(name: str):
        return create_qnet(
            action_space,
            observation_space,
            name,
            conf
        )

    return EnsembleAgent(
        qnet_constructor,
        action_space,
        exploration_schedule,
        conf,
    )
