""" agents that learn a Q function directly """

from collections import deque
from functools import partial
from typing import Callable, Deque
import numpy as np

from mypy_extensions import NamedArg

from po_nrl.agents.agent import Agent, RandomAgent
from po_nrl.agents.misc import ExplorationSchedule, FixedExploration
from po_nrl.agents.misc import epsilon_greedy, PiecewiseSchedule
from po_nrl.agents.neural_networks import create_qnet
from po_nrl.agents.neural_networks.q_functions import QNetInterface
from po_nrl.environments import ActionSpace
from po_nrl.misc import Space, POBNRLogger


class BaselineAgent(Agent, POBNRLogger):
    """ default single q-net agent implementation"""

    def __init__(self,
                 qnet: QNetInterface,
                 action_space: ActionSpace,
                 exploration: ExplorationSchedule,
                 conf):
        """ initializes a single network

        Args:
             qnet: (`pobnrl.agents.neural_networks.q_functions.QNetInterface`): \
                    Q-net to use
             action_space: (`pobnrl.environments.ActionSpace`): of environment
             exploration: (`pobnrl.agents.misc.ExplorationSchedule`): \
                    schedule for e-greedy
             conf: (`namespace`) set of configurations (see -h)

        """

        POBNRLogger.__init__(self)

        assert conf.target_update_freq > 0
        assert conf.train_freq > 0
        assert conf.history_len > 0

        # params
        self.target_update_freq = conf.target_update_freq
        self.train_freq = conf.train_freq

        self.action_space = action_space

        self.timestep = 0
        self.last_action = -1
        self.last_obs: Deque[np.ndarray] = deque([], conf.history_len)

        self.exploration = exploration

        self.q_net = qnet

    def reset(self) -> None:
        """ re-initializes members

        e.g.  empties replay buffer and sets timestep to 0, resets net
        """

        self.timestep = 0
        self.last_action = -1
        self.last_obs.clear()

        self.q_net.reset()

    def episode_reset(self, observation: np.ndarray) -> None:
        """ prepares for next episode

        stores the observation and resets its Q-network

        Args:
             observation: (`np.ndarray`): stored and later given to qnet

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
        self.log(POBNRLogger.LogLevel.V4, f"Agent's epsilon is {epsilon}")

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
            self.last_obs[-1], self.last_action, reward, observation, terminal
        )

        self.last_obs.append(observation)

        if self.timestep % self.train_freq == 0:
            self.q_net.batch_update()

        if self.timestep % self.target_update_freq == 0:
            self.q_net.update_target()

        self.timestep += 1


class EnsembleAgent(Agent, POBNRLogger):
    """ ensemble agent """

    def __init__(self,
                 qnet_constructor: Callable[[NamedArg(str, 'name')], QNetInterface],
                 action_space: ActionSpace,
                 exploration: ExplorationSchedule,
                 conf):
        """ initialize network

        Args:
            qnet_constructor: (`Callable`[[], `pobnrl.agents.neural_networks.q_functions.QNetInterface`]): \
                    Q-net constructor to use to create nets (given scope)
            action_space: (`pobnrl.environments.ActionSpace`): of environment
            exploration: (`pobnrl.agents.misc.ExplorationSchedule`): \
                    exploration schedule
            conf: (`namespace`): set of configurations

        """

        POBNRLogger.__init__(self)

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
        self.last_action = -1
        self.last_obs: Deque[np.ndarray] = deque([], conf.history_len)

        self.nets = np.array([qnet_constructor(name=f'net {i}') for i in range(conf.num_nets)])

        self._storing_nets = self.nets[np.random.rand(len(self.nets)) > .5]
        self._current_policy = np.random.choice(self.nets)

    def reset(self) -> None:
        """ re-initializes members

        e.g.  empties replay buffer and sets timestep to 0, resets nets
        """

        self.timestep = 0
        self.last_action = -1
        self.last_obs.clear()

        self._storing_nets = self.nets[np.random.rand(len(self.nets)) > .5]
        self._current_policy = np.random.choice(self.nets)

        for net in self.nets:
            net.reset()

        self.log(
            POBNRLogger.LogLevel.V3,
            f"Ensem Agent resets: current policy is {self._current_policy}"
            f" and {self._storing_nets} are currently active"
        )

    def episode_reset(self, observation: np.ndarray) -> None:
        """ prepares for next episode

        stores the observation and resets its Q-network and resets its models

        Args:
             observation: (`np.ndarray`):  stored and later given to qnet

        """

        self.last_obs.clear()
        self.last_obs.append(observation)

        self._storing_nets = self.nets[np.random.rand(len(self.nets)) > .5]
        self._current_policy = np.random.choice(self.nets)

        for net in self.nets:
            net.episode_reset()

        self.log(
            POBNRLogger.LogLevel.V3,
            f"Ensem Agent preps for new episode: "
            f"current policy is {self._current_policy}"
            f" and {self._storing_nets} are currently active"
        )

    def select_action(self) -> int:
        """ returns greedy action from current active policy

        RETURNS (`int`):

        """

        q_values = self._current_policy.qvalues(np.array(self.last_obs))

        epsilon = self.exploration.value(self.timestep)

        self.log(
            POBNRLogger.LogLevel.V4,
            f"Ensemble Agent's epsilon is {epsilon}"
        )

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
                self.last_obs[-1], self.last_action, reward, observation, terminal
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
        action_space: ActionSpace,
        observation_space: Space,
        conf) -> Agent:
    """ factory function to construct model-free learning agents

    Args:
         action_space: (`pobnrl.environments.ActionSpace`): of environment
         observation_space: (`pobnrl.misc.Space`) of environment
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.agent.Agent`)

    """

    if conf.random_policy:
        return RandomAgent(action_space)

    if 0 <= conf.exploration <= 1:
        exploration_schedule: ExplorationSchedule = FixedExploration(conf.exploration)
    else:
        exploration_schedule = PiecewiseSchedule(
            [(0, 1.0), (2e4, 0.1), (1e5, 0.05)], outside_value=0.05
        )

    # single-net agent
    if conf.num_nets == 1:
        return BaselineAgent(
            create_qnet(
                action_space=action_space,
                observation_space=observation_space,
                conf=conf,
                name='net'
            ),
            action_space,
            exploration_schedule,
            conf
        )

    # num_nets > 1: ensemble agent
    return EnsembleAgent(
        partial(
            create_qnet,
            action_space=action_space,
            observation_space=observation_space,
            conf=conf
        ),
        action_space,
        exploration_schedule,
        conf,
    )
