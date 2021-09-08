"""Describes the protocol for a domain in this package"""
import numpy as np
from typing_extensions import Protocol

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
)
from general_bayes_adaptive_pomdps.misc import DiscreteSpace


class Domain(Protocol):
    """The protocol for a domain in this package.

    This is not necessary for any particular reason, other than that it makes
    it simpler to implement :class:`GBAPOMDP` and similar classes when one can
    assume some functionality. So if this protocol is implemented, it can be
    used as a domain to do GBA-POMDP on.
    """

    @property
    def state_space(self) -> DiscreteSpace:
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
    def observation_space(self) -> DiscreteSpace:
        """returns size of domain observation space

        RETURNS(`general_bayes_adaptive_pomdps.misc.DiscreteSpace`): the observation space

        """

    def reset(self) -> np.ndarray:
        """resets internal state and return first observation"""

    def step(self, action: int) -> DomainStepResult:
        """update state as a result of action

        May raise :class:`TerminalState`

        Args:
             action: (`int`): agent's taken action

        RETURNS (`EnvironmentInteraction`): the transition

        """

    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """generates a transition

        May raise :class:`TerminalState`

        Args:
             state: (`np.ndarray`): some state
             action: (`int`): agent's taken action

        RETURNS (`SimulationResult`): the transition

        """

    def sample_start_state(self) -> np.ndarray:
        """returns a potential start state"""

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


class DomainPrior(Protocol):
    """The interface to priors"""

    def sample(self) -> Domain:
        """sample a simulator

        RETURNS (`Domain`):
        """
