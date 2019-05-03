""" environment interface """

from collections import namedtuple
from typing import Any

import abc
import numpy as np

from pobnrl.misc import DiscreteSpace

from .misc import ActionSpace


class EnvironmentInteraction(
        namedtuple('environment_interaction', 'observation reward terminal')):
    """ The tuple returned by environments doing steps

        Contains:
             observation: (`np.ndarray`)
             reward: (`float`)
             terminal: (`bool`)

    """

    __slots__ = ()  # required to keep lightweight implementation of namedtuple


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

        RETURNS (`EnvironmentInteraction`): the transition

        """

    @property
    @abc.abstractmethod
    def action_space(self) -> ActionSpace:
        """ returns size of domain action space

        TODO: update to ActionSpace

        RETURNS(`pobnrl.environments.misc.ActionSpace`): the action space

        """

    @property
    @abc.abstractmethod
    def observation_space(self) -> DiscreteSpace:
        """ returns size of domain observation space

        RETURNS(`pobnrl.misc.DiscreteSpace`): the observation space

        """

    def __repr__(self):
        return (f"{self.__class__} with action space {self.action_space}, "
                f"observation space {self.observation_space}")


class SimulatedInteraction(
        namedtuple('simulated_interaction', 'state observation reward terminal')):
    """ The tuple returned by simulations doing steps

        Contains:
             state: (`Any`)
             observation: (`np.ndarray`)
             reward: (`float`)
             terminal: (`bool`)

    """

    __slots__ = ()  # required to keep lightweight implementation of namedtuple


class Simulator(abc.ABC):
    """ generative environment interface

    TODO: add reward(s,a,s') and terminal(s,a,s')

    """

    @abc.abstractmethod
    def simulation_step(self, state: Any, action: int) -> SimulatedInteraction:
        """ generates a transition

        Args:
             state: (`Any`): some state
             action: (`int`): agent's taken action

        RETURNS (`SimulatedInteraction`): the transition

        """

    @abc.abstractmethod
    def sample_start_state(self) -> Any:
        """ returns a potential start state """

    @abc.abstractmethod
    def obs2index(self, observation: np.array) -> int:
        """ projects the observation as an int

        Args:
             observation: (`np.array`): observation to project

        RETURNS (`int`): int representation of observation

        """
