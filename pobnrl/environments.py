"""environments interface """

from collections import namedtuple
from typing import Any
import abc
import numpy as np

from misc import DiscreteSpace


class ActionSpace(DiscreteSpace):
    """ action space forenvironmentss

    TODO: add one_hot function

    """

    def __init__(self, size: int):
        """ initiates an action space of size

        Args:
             dim: (`int`): number of actions

        """
        super().__init__([size])

    def sample(self) -> np.array:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return super().sample()[0]

    def __repr__(self):
        return f"ActionSpace of size {self.n}"

# TODO: implement gymspace to allow cartpole to work again
# class GymSpace(DiscreteSpace):
    # """ wrapper for open ai gym discrete spaces """


class EnvironmentInteraction(
        namedtuple('environments_interaction', 'observation reward terminal')):
    """ The tuple returned by domains doing steps

        Contains:
             observation: (`np.ndarray`)
             reward: (`float`)
             terminal: (`bool`)

    """

    __slots__ = ()  # required to keep lightweight implementation of namedtuple


class Environment(abc.ABC):
    """ interface to all domains """

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

        RETURNS(`pobnrl.environments.ActionSpace`): the action space

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


class POUCTInteraction(
        namedtuple('simulated_interaction', 'state observation reward terminal')):
    """ The tuple returned by simulations doing steps

        Contains:
             state: (`Any`)
             observation: (`np.ndarray`)
             reward: (`float`)
             terminal: (`bool`)

    """

    __slots__ = ()  # required to keep lightweight implementation of namedtuple


class POUCTSimulator(abc.ABC):
    """ generative environment interface """

    @property
    @abc.abstractmethod
    def action_space(self) -> ActionSpace:
        """ returns size of domain action space

        RETURNS(`pobnrl.environments.ActionSpace`): the action space

        """

    @abc.abstractmethod
    def simulation_step(self, state: Any, action: int) -> POUCTInteraction:
        """ generates a transition

        Args:
             state: (`Any`): some state
             action: (`int`): agent's taken action

        RETURNS (`pobnrl.environments.POUCTInteraction`): the transition

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
