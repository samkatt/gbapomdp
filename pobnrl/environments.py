"""environments interface """

from collections import namedtuple
import abc
import gym
import numpy as np

from misc import Space, DiscreteSpace


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

    def contains(self, action: int) -> bool:
        """ returns whether `this` contains action

        Args:
             action: (`int`): an action

        RETURNS (`bool`): true if in `this`

        """
        return super().contains(np.array([action]))


class GymSpace(Space):
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
    def reset(self) -> np.ndarray:
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
    def observation_space(self) -> Space:
        """ returns size of domain observation space

        RETURNS(`pobnrl.misc.DiscreteSpace`): the observation space

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

        RETURNS (`pobnrl.misc.DiscreteSpace`):

        """

    @property
    @abc.abstractmethod
    def action_space(self) -> ActionSpace:
        """ returns size of domain action space

        RETURNS(`pobnrl.environments.ActionSpace`): the action space

        """

    @property
    @abc.abstractmethod
    def observation_space(self) -> Space:
        """ returns size of domain observation space

        RETURNS(`pobnrl.misc.DiscreteSpace`): the observation space

        """

    @abc.abstractmethod
    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """ generates a transition

        Args:
             state: (`np.ndarray`): some state
             action: (`int`): agent's taken action

        RETURNS (`SimulationResult`): the transition

        """

    @abc.abstractmethod
    def sample_start_state(self) -> np.ndarray:
        """ returns a potential start state """

    @abc.abstractmethod
    def obs2index(self, observation: np.ndarray) -> int:
        """ projects the observation as an int

        Args:
             observation: (`np.ndarray`): observation to project

        RETURNS (`int`): int representation of observation

        """

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
