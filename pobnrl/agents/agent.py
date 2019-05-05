""" agent implementation and some simple baselines """

import abc
import numpy as np

from environments import ActionSpace


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
    def select_action(self) -> int:
        """ asks the agent to select an action

        RETURNS (`int`):

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

    def __repr__(self):
        return f"{self.__class__}"


class RandomAgent(Agent):
    """ Acts randomly """

    def __init__(self, action_space: ActionSpace):
        """ constructs an agent that will act randomly

        Args:
             num_actions: (`pobnrl.environments.ActionSpace`): the action space

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

    def select_action(self) -> int:
        """ returns a random action

        Part of the interface of `pobnrl.agents.agent.Agent`

        RETURNS (`int`):

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
