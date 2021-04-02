"""Core functionality and models

Contains the protocol for the GBA-POMDP and its domains. Additionally provides
building block classes used in the rest of the code.

"""

from typing import Any, NamedTuple, TypeVar

import numpy as np
from typing_extensions import Protocol

from general_bayes_adaptive_pomdps.misc import DiscreteSpace, Space


class ActionSpace(DiscreteSpace):
    """action space for domains"""

    def __init__(self, size: int):
        """initiates an action space of size

        Args:
             dim: (`int`): number of actions

        """
        super().__init__([size])

    def sample(self) -> np.ndarray:
        """returns a sample from the space at random

        NOTE: implements :class:`DiscreteSpace` protocol and thus returns an
        array of 1 element. See :meth:`sample_as_int` to directly sample as
        integer

        RETURNS (`np.array`): a sample in the space of this

        """
        return super().sample()

    def one_hot(self, action: int) -> np.ndarray:
        """returns a 1-hot encoding of the action

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
        """returns whether `this` contains action

        Args:
             elem: (`int`): an action

        RETURNS (`bool`): true if in `this`

        """
        return super().contains(np.array([elem]))

    def sample_as_int(self) -> int:
        """Samples an action in its int representation"""
        return int(self.sample()[0])


class DomainStepResult(NamedTuple):
    """the tuple returned by domains doing steps"""

    observation: np.ndarray
    reward: float
    terminal: bool


class TerminalState(Exception):
    """ raised when trying to step with a terminal state """


class SimulationResult(NamedTuple):
    """The tuple returned by simulations doing steps """

    state: Any
    observation: np.ndarray


class Domain(Protocol):
    """The protocol for a domain in this package.

    This is not necessary for any particular reason, other than that it makes
    it simpler to implmeent :class:`GBAPOMDP` and similar classes when one can
    assume some functionality. So if this protocol is implemented, it can be
    used as a domain to do GBA-POMDP on.
    """

    @property
    def state_space(self) -> Space:
        """the (discrete) state space of the POMDP

        Args:

        RETURNS (`general_bayes_adaptive_pomdps.misc.DiscreteSpace`):

        """

    @property
    def action_space(self) -> ActionSpace:
        """returns size of domain action space

        RETURNS(`general_bayes_adaptive_pomdps.core.ActionSpace`): the action space

        """

    @property
    def observation_space(self) -> Space:
        """returns size of domain observation space

        RETURNS(`general_bayes_adaptive_pomdps.misc.DiscreteSpace`): the observation space

        """

    def reset(self) -> np.ndarray:
        """ resets internal state and return first observation """

    def step(self, action: int) -> DomainStepResult:
        """update state as a result of action

        May raise `TerminalState`

        Args:
             action: (`int`): agent's taken action

        RETURNS (`EnvironmentInteraction`): the transition

        """

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """generates a transition

        May raise `TerminalState`

        Args:
             state: (`np.ndarray`): some state
             action: (`int`): agent's taken action

        RETURNS (`SimulationResult`): the transition

        """

    def sample_start_state(self) -> np.ndarray:
        """ returns a potential start state """

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """the reward function

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """the termination function

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`): whether the transition is terminal

        """

    def state_to_string(self, state: np.ndarray) -> str:
        """Returns a string representation of the `state`

        Exists so that derived classes can override this, and hopefully provide
        more useful info than the array representation

        Args:
            state (`np.ndarray`):

        Returns:
            `str`:
        """
        return str(state)

    def action_to_string(self, action: int) -> str:
        """Returns a string representation of the `action`

        Exists so that derived classes can override this, and hopefully provide
        more useful info than the int representation

        Args:
            action (`int`):

        Returns:
            `str`:
        """
        return str(action)

    def observation_to_string(self, observation: np.ndarray) -> str:
        """Returns a string representation of the `observation`

        Exists so that derived classes can override this, and hopefully provide
        more useful info than the array representation

        Args:
            observation (`np.ndarray`):

        Returns:
            `str`:
        """
        return str(observation)


AugmentedState = TypeVar("AugmentedState")
"""The state space of a General Bayes-adaptive POMDP"""


class GeneralBAPOMDP(Protocol[AugmentedState]):
    """Defines the protocol of a general Bayes-adaptive POMDP

    The GBA-POMDP **is a** POMDP with a special state space. It augments an
    existing POMDP, of which the dynamics are unknown, and constructs a larger
    POMDP with known dynamics. Here we describe the generally assumed API.

    The class is templated by the state space
    """

    @property
    def action_space(self) -> ActionSpace:
        """The space of legal actions in this GBA-POMDP

        This space should be the same as the action space of the underlying POMDP

        :return: legal actions in ``self``
        """

    @property
    def observation_space(self) -> DiscreteSpace:
        """The space of possible observations in this GBA-POMDP

        This space should be the same as the observation space of the underlying POMDP

        Note that this space may be much larger than the actual observations
        found during interactions in the (GBA-) POMDP.

        :return: space of possible observations
        """

    def sample_start_state(self) -> AugmentedState:
        """Basic functionality of POMDPs: returns an initial state

        This typical is some combination of a prior belief over the model of
        the underlying POMDP and an initial state in the underlying POMDP.

        :return: initial state in the GBA-POMDP
        """
        ...

    def simulation_step(
        self, state: AugmentedState, action: int, optimize: bool = False
    ) -> SimulationResult:
        """Performs an actual step according to the GBA-POMDP dynamics

        The resulting `SimulationResult` contains the generated state and observation, where the state is generated by:

            1. sample a new domain state and observation according to the belief over the model in ``state``
            2. update model in ``state`` according to sampled transition

        Note that this operation is expensive, as it often involves generating
        a new set of parameters to represent the model distribution in the new
        state (which involves a copy before updating). If there is no need for
        the input ``state`` to stay unmodified, consider calling
        `domain_simulation_step` and `model_simulation_step` to achieve the
        same without the copy.

        :param state: state at timestep t
        :param action: action at timestep t
        :param optimize: optimization flag.
            If set to true, the model in ``state`` is _not_ copied, and thus
            the incoming ``state`` **is modified**. If there is no need to keep
            the old model, then setting this flag skips a then needless copy
            operation, which can be significant
        :return: (state, obs) at timestep t+1
        """

    def domain_simulation_step(
        self, state: AugmentedState, action: int
    ) -> SimulationResult:
        """Performs the domain state part of the step in the GBA-POMDP

        The typical step updates both the state and the model. For some
        operations it is useful to break up these two steps. This function
        contains the first step. It will sample a next state and observation
        according to the model distribution in ``state`` and return those.

        NOTE: that the model in the returned state is a reference to the one in
        ``state`` so be careful

        :param state: state at timestep t
        :param action: aciton at timestep t
        :return: (state, obs) at t+1, where model in state == of the incoming ``state``
        """

    def model_simulation_step(
        self,
        to_update: AugmentedState,
        prev_state: AugmentedState,
        action: int,
        next_state: AugmentedState,
        obs: np.ndarray,
        optimize: bool = False,
    ) -> AugmentedState:
        """Performs the model part of the step in the GBA-POMDP

        The typical step updates both the state and the model. For some
        operations it is useful to break up these two steps. This function
        contains the second step. It will update the model in ``to_update``
        according to the transition (``prev_state``, ``action``,
        ``next_state``, ``obs``) and return a new augmented state with
        that model.

        NOTE: the domain state in the returned state is a reference to the one
        in ``to_update`` so be careful.

        :param to_update: the augmented state that contains the model at timestep t
        :param prev_state: the state at timestep t of the transition
        :param action: the action at timestep t of the transition
        :param next_state: the state at timestep t+1 of the transition
        :param obs: the observation at timestep t of the transition
        :param optimize: optimization flag.
            If set to true, the model in ``to_update`` is _not_ copied, and
            thus the incoming ``to_update`` **is modified**. If there is no
            need to keep the old model, then setting this flag skips a then
            needless copy operation, which can be significant
        :return: new state with updated model (same domain state as ``to_update``)
        """

    def reward(
        self, state: AugmentedState, action: int, next_state: AugmentedState
    ) -> float:
        """computes the reward associated with transition

        The reward function, typically calls the (known) reward function of the
        underlying POMDP with the domain states in the (s,a,s') transition

        :param state: state at timestep t
        :param action: action at timestep t
        :param next_state: state at timestep t+1
        :return: reward generated in input transition
        """

    def terminal(
        self, state: AugmentedState, action: int, next_state: AugmentedState
    ) -> bool:
        """computes whether the input transition is terminal

        The terminality function, typically calls the (known) terminality
        function of the underlying POMDP with the domain states in the
        (s,a,s') transition

        :param state: state at timestep t
        :param action: action at timestep t
        :param next_state: state at timestep t+1
        :return: true is the input transition was terminal
        """


class DomainPrior(Protocol):
    """The interface to priors"""

    def sample(self) -> Domain:
        """sample a simulator

        RETURNS (`general_bayes_adaptive_pomdps.core.Domain`):
        """
