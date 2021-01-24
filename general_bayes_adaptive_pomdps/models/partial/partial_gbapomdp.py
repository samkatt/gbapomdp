"""Structures that help define and construct partial GBA-POMDPs

To prevent re-inventing the wheel too much, this module contains structures
that should simplify creating GBA-POMDPs:

    - :class:`GBAPOMDPThroughAugmentedState`
"""
from copy import deepcopy
from typing import Callable

import numpy as np
from typing_extensions import Protocol

from general_bayes_adaptive_pomdps.domains.gbapomdp import GBAPOMDP
from general_bayes_adaptive_pomdps.environments import (
    ActionSpace,
    DiscreteSpace,
    SimulationResult,
)


class AugmentedGodState(Protocol):
    """The protocol for an augmented state that implements all the details of a GBA-POMDP"""

    def update_theta(
        self,
        state: "AugmentedGodState",
        action: int,
        next_state: "AugmentedGodState",
        obs: np.ndarray,
    ):
        """Updates the model posterior of ``self`` given (``state``, ``action``, ``next_state``, ``obs``

        Given the transition (s,a,s',o), this function updates the model parameters

        NOTE: modifies ``self``

        :param state:
        :param action:
        :param next_state:
        :param obs:
        """

    def update_domain_state(self, action: int) -> np.ndarray:
        """Updates _in place_ the domain state under ``action`` and generate observation

        NOTE: modifies ``self``

        :param action:
        :return: the observation generated during the domain state update
        """

    def reward(self, action: int, next_state: "AugmentedGodState") -> float:
        """the reward associated with ``self`` transitioning to ``next_state`` under ``action``

        :param action:
        :param next_state:
        :return: the reward for the transition (self, action, next_state)
        """

    def terminal(self, action: int, next_state: "AugmentedGodState") -> bool:
        """the terminality associated with ``self`` transitioning to ``next_state`` under ``action``

        :param action:
        :param next_state:
        :return: true if the transition (self, action, next_state) is terminal
        """


class Prior(Protocol):
    """The interface for sampling initial states for the partial GBA-POMDP"""

    def __call__(self) -> AugmentedGodState:
        """The prior is defined by a call to sample initial states

        :return: An initial :class:`AugmentedGodState`
        """


class GBAPOMDPThroughAugmentedState(GBAPOMDP[AugmentedGodState]):
    """A GBA-POMDP whose implementation is provided by the :class:`AugmentedGodState`

    The idea of this framework is to exploit the protocol of the
    :class:`AugmentedGodState`, called "God" for the disproportional amount of
    functionality it can contains. This approach wraps such a powerful
    augmented state, and adds some boiletplate code to implement the
    :class:`~general_bayes_adaptive_pomdps.gbapomdp.GBAPOMDP` protocol.
    """

    def __init__(
        self,
        prior: Callable[[], AugmentedGodState],
        action_space: ActionSpace,
        obs_space: DiscreteSpace,
    ):
        """Creates a GBA-POMDP given the ``prior``

        Most of the functionality of this GBA-POMDP is (assumed) implemented in
        :class:`AugmentedGodState`. Sampling start states is done through
        ``prior``, and all the dynamics are done by calling the input states.

        :param prior: A function that returns an initial GBA-POMDP state
        :param action_space:
        :param obs_space:
        """
        super().__init__()
        self._action_space = action_space
        self._obs_space = obs_space
        self.prior = prior

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        return self._obs_space

    def sample_start_state(self) -> AugmentedGodState:
        return self.prior()

    def simulation_step_inplace(
        self, state: AugmentedGodState, action: int
    ) -> SimulationResult:
        """Performs simulation step

        NOTE: modifies incoming ``state``

        Simulates performing an ``action`` in ``state`` of the GBA-POMDP, as
        implemented by :class:`AugmentedGodState`:

            - :meth:`AugmentedGodState.update_domain_state`
            - :meth:`AugmentedGodState.update_theta`

        and returns the next state with generated observation

        :param state:
        :param action:
        :return: (next_state, observation)
        """
        obs = state.update_domain_state(action)
        state.update_theta(state, action, state, obs)

        return SimulationResult(state, obs)

    def simulation_step(
        self, state: AugmentedGodState, action: int
    ) -> SimulationResult:
        """Performs a simulation in the GBA-POMDP

        NOTE: calls :meth:`simulation_step_inplace` with a copy of ``state``

        :param state:
        :param action:
        :return: (next_state, observation)
        """

        # ensure ``state`` unmodified
        next_state = deepcopy(state)

        return self.simulation_step_inplace(next_state, action)

    def reward(
        self, state: AugmentedGodState, action: int, next_state: AugmentedGodState
    ) -> float:
        return state.reward(action, next_state)

    def terminal(
        self, state: AugmentedGodState, action: int, next_state: AugmentedGodState
    ) -> bool:
        return state.terminal(action, next_state)
