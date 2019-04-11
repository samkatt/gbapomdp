""" environment interface """

import abc
import numpy as np

from misc import DiscreteSpace


class EnvironmentInteraction:  # pylint: disable=too-few-public-methods
    """ a step in an environment """

    def __init__(
            self,
            observation: np.array,
            reward: float,
            terminal: bool):
        """ __init__

        Args:
             observation: (`np.array`):
             reward: (`float`):
             terminal: (`bool`):

        """
        self.observation = observation
        self.terminal = terminal
        self.reward = reward


class Environment(abc.ABC):
    """ interface to all environments """

    @abc.abstractmethod
    def reset(self):
        """ resets internal state and return first observation """

    @abc.abstractmethod
    def step(self, action: int) -> EnvironmentInteraction:
        """ update state as a result of action

        Args:
             action: (`int`): agent's taken action

        RETURNS (`pobnrl.environments.environment.EnvironmentInteraction`): the transition

        """

    @property
    @abc.abstractmethod
    def state(self):
        """ returns the current state """

    @property
    @abc.abstractmethod
    def action_space(self) -> DiscreteSpace:
        """ returns size of domain action space

        RETURNS(`pobnrl.misc.DiscreteSpace`): the action space

        """

    @property
    @abc.abstractmethod
    def observation_space(self) -> DiscreteSpace:
        """ returns size of domain observation space

        RETURNS(`pobnrl.misc.DiscreteSpace`): the observation space

        """

    def __repr__(self):
        return (f"{self.__class__} with action space {self.action_space}, "
                f"observation space {self.observation_space} and "
                f"current state {self.state}")
