"""Structures that help define and construct partial GBA-POMDPs

To prevent re-inventing the wheel too much, this module contains structures
that should simplify creating GBA-POMDPs:

    - :class:`GBAPOMDPThroughAugmentedState`
"""
from typing import Tuple

import numpy as np
from typing_extensions import Protocol

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DiscreteSpace,
    GeneralBAPOMDP,
    SimulationResult,
)


class AugmentedGodState(Protocol):
    """The protocol for an augmented state that implements all the details of a GBA-POMDP"""

    def update_model_distribution(
        self,
        state: "AugmentedGodState",
        action: int,
        next_state: "AugmentedGodState",
        obs: np.ndarray,
        optimize: bool = False,
    ) -> "AugmentedGodState":
        """Updates the model posterior of ``self`` given (``state``, ``action``, ``next_state``, ``obs``

        Given the transition (s,a,s',o), this function updates the model parameters

        NOTE: modifies ``self`` if ``optimize`` is set to true

        :param state:
        :param action:
        :param next_state:
        :param obs:
        :param optimize: optimization flag.
            If set to true, the model in ``self`` is _not_ copied, and thus
            ``self`` **is modified**. If there is no need to keep the old
            model, then setting this flag skips a then needless copy operation,
            which can be significant
        :return: A new state with updated model parameters
        """

    def domain_step(self, action: int) -> Tuple["AugmentedGodState", np.ndarray]:
        """Simulates a step in the domain

        Samples a next state (from ``self``) and observation given input
        ``action``. The result is put into a new ``AugmentedGodState``, with
        **unmodified model distribution**. The new state and observation is
        returned

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


class GBAPOMDPThroughAugmentedState(GeneralBAPOMDP[AugmentedGodState]):
    """A GBA-POMDP whose implementation is provided by the :class:`AugmentedGodState`

    The idea of this framework is to exploit the protocol of the
    :class:`AugmentedGodState`, called "God" for the disproportional amount of
    functionality it can contains. This approach wraps such a powerful
    augmented state, and adds some boiletplate code to implement the
    :class:`~general_bayes_adaptive_pomdps.gbapomdp.GBAPOMDP` protocol.
    """

    def __init__(
        self,
        prior: Prior,
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

    def simulation_step(
        self, state: AugmentedGodState, action: int, optimize: bool = False
    ) -> SimulationResult:
        """Performs an actual step according to the GBA-POMDP dynamics

        Part of GBAPOMDP protocol
        """

        next_state, obs = state.domain_step(action)
        next_state = next_state.update_model_distribution(
            state, action, next_state, obs, optimize
        )

        return SimulationResult(next_state, obs)

    def domain_simulation_step(
        self, state: AugmentedGodState, action: int
    ) -> SimulationResult:
        """Performs the domain state part of the step in the GBA-POMDP

        Part of GBAPOMDP protocol

        """
        return SimulationResult(*state.domain_step(action))

    def model_simulation_step(
        self,
        to_update: AugmentedGodState,
        prev_state: AugmentedGodState,
        action: int,
        next_state: AugmentedGodState,
        obs: np.ndarray,
        optimize: bool = False,
    ) -> AugmentedGodState:
        """Performs the model part of the step in the GBA-POMDP

        Part of GBAPOMDP protocol

        """
        return to_update.update_model_distribution(
            prev_state, action, next_state, obs, optimize
        )

    def reward(
        self, state: AugmentedGodState, action: int, next_state: AugmentedGodState
    ) -> float:
        return state.reward(action, next_state)

    def terminal(
        self, state: AugmentedGodState, action: int, next_state: AugmentedGodState
    ) -> bool:
        return state.terminal(action, next_state)
