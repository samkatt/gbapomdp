"""environments interface """

from collections import namedtuple
from typing import Any
import abc
import gym
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


class GymSpace():
    """ wrapper for open ai gym spaces """

    def __init__(self, wrapped_space: gym.Space):

        assert len(wrapped_space.shape) == 1, "only support 1-dim spaces for now"

        self._wrapped_space = wrapped_space

    @property
    def n(self) -> int:  # pylint: disable=invalid-name
        """ Number of elements in space

        While the naming is pretty awful, it is consistent with the `Space`
        class of open AI gym, which I prioritized here

        RETURNS (`int`):

        """
        return self._wrapped_space.n

    @property
    def ndim(self) -> int:
        """ returns the numbe of dimensions

        returns the number of elements according to shape of wrapped space

        RETURNS (`int`): number of dimensions

        """
        return self._wrapped_space.shape[0]

    def contains(self, elem: np.ndarray) -> bool:
        """ returns `this` contains `elem`

        Directly calls the wrapped space

        Args:
             elem: (`np.ndarray`): element to check against

        RETURNS (`bool`):

        """

        return self._wrapped_space.contains(elem)

    def sample(self) -> np.array:
        """ returns a sample from the space at random

        Directly calls the wrapped space

        RETURNS (`np.array`): a sample in the space of this

        """
        return self._wrapped_space.sample()

    def __repr__(self):
        return f"Wrapped gym space {self._wrapped_space}"


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
