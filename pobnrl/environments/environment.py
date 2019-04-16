""" environment interface """

from collections import namedtuple
from typing import Any

import abc
import numpy as np

from misc import DiscreteSpace


# TODO: cannot parse source with `./make_documentation`
# TODO: remove state and create a new type to include state?
EnvironmentInteraction = namedtuple(
    'EnvironmentInteraction',
    'state observation reward terminal'
)


# TODO: slim down this interface
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
    def state(self):
        """ returns the current state """

    @state.setter
    @abc.abstractmethod
    def state(self, state: Any):
        """ sets state """

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

    def __repr__(self):
        return (f"{self.__class__} with action space {self.action_space}, "
                f"observation space {self.observation_space} and "
                f"current state {self.state}")
