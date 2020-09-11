"""environments interface """

from enum import Enum

from collections import namedtuple
import abc
import numpy as np

from po_nrl.misc import Space, DiscreteSpace


class EncodeType(Enum):
    """ AgentType """
    DEFAULT = 0
    ONE_HOT = 1


class ActionSpace(DiscreteSpace):
    """ action space forenvironmentss """

    def __init__(self, size: int):
        """ initiates an action space of size

        Args:
             dim: (`int`): number of actions

        """
        super().__init__([size])

    def sample(self) -> np.ndarray:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return super().sample()[0]

    def one_hot(self, action: int) -> np.ndarray:
        """ returns a 1-hot encoding of the action

        Args:
             action: (`int`):

        RETURNS (`np.ndarray`):

        """

        assert self.contains(action)

        one_hot_rep = np.zeros(self.size)
        one_hot_rep[action] = 1

        return one_hot_rep

    def __repr__(self):
        return f"ActionSpace of size {self.n}"

    def contains(self, elem: int) -> bool:
        """ returns whether `this` contains action

        Args:
             elem: (`int`): an action

        RETURNS (`bool`): true if in `this`

        """
        return super().contains(np.array([elem]))


class EnvironmentInteraction(
        namedtuple('environments_interaction', 'observation reward terminal')):
    """ The tuple returned by domains doing steps

        Contains:
             observation: (`np.ndarray`)
             reward: (`float`)
             terminal: (`bool`)

    """

    __slots__ = ()  # required to keep lightweight implementation of namedtuple


class TerminalState(Exception):
    """ raised when trying to step with a terminal state """


class Environment(abc.ABC):
    """ interface to all domains """

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """ resets internal state and return first observation """

    @abc.abstractmethod
    def step(self, action: int) -> EnvironmentInteraction:
        """ update state as a result of action

        May raise `TerminalState`

        Args:
             action: (`int`): agent's taken action

        RETURNS (`EnvironmentInteraction`): the transition

        """

    @property
    @abc.abstractmethod
    def action_space(self) -> ActionSpace:
        """ returns size of domain action space

        RETURNS(`po_nrl.environments.ActionSpace`): the action space

        """

    @property
    @abc.abstractmethod
    def observation_space(self) -> Space:
        """ returns size of domain observation space

        RETURNS(`po_nrl.misc.DiscreteSpace`): the observation space

        """

    def __repr__(self):
        return (f"{self.__class__} with action space {self.action_space}, "
                f"observation space {self.observation_space}")


class SimulationResult(
        namedtuple('simulated_interaction', 'state observation')):
    """ The tuple returned by simulations doing steps

        Contains:
             state: (`Any`)
             observation: (`np.ndarray`)

    """

    # required to keep lightweight implementation of namedtuple
    __slots__ = ()


class Simulator(abc.ABC):
    """ generative environment interface """

    @property
    @abc.abstractmethod
    def state_space(self) -> Space:
        """ the (discrete) state space of the POMDP

        Args:

        RETURNS (`po_nrl.misc.DiscreteSpace`):

        """

    @property
    @abc.abstractmethod
    def action_space(self) -> ActionSpace:
        """ returns size of domain action space

        RETURNS(`po_nrl.environments.ActionSpace`): the action space

        """

    @property
    @abc.abstractmethod
    def observation_space(self) -> Space:
        """ returns size of domain observation space

        RETURNS(`po_nrl.misc.DiscreteSpace`): the observation space

        """

    @abc.abstractmethod
    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """ generates a transition

        May raise `TerminalState`

        Args:
             state: (`np.ndarray`): some state
             action: (`int`): agent's taken action

        RETURNS (`SimulationResult`): the transition

        """

    @abc.abstractmethod
    def sample_start_state(self) -> np.ndarray:
        """ returns a potential start state """

    @abc.abstractmethod
    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """ the reward function

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """

    @abc.abstractmethod
    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """ the termination function

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`): whether the transition is terminal

        """
