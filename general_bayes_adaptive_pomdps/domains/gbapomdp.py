"""The definition of a general Bayes adaptive POMDP

Provides the protocol
"""

from typing import TypeVar

from typing_extensions import Protocol

from general_bayes_adaptive_pomdps.environments import ActionSpace, SimulationResult
from general_bayes_adaptive_pomdps.misc import DiscreteSpace

AugmentedState = TypeVar("AugmentedState")
"""The state space of a General Bayes-adaptive POMDP"""


class GBAPOMDP(Protocol[AugmentedState]):
    """Defines the protocol of a general Bayes-adaptive POMDP

    The GBA-POMDP _is a_ POMDP with a special state space. It augments an
    existing POMDP, of which the dynamics are unknown, and constructs a larger
    POMDP with known dynamics. Here we describe the generally assumed API.

    The class is templated by the state space
    """

    @property
    def action_space(self) -> ActionSpace:
        ...

    @property
    def observation_space(self) -> DiscreteSpace:
        ...

    def sample_start_state(self) -> AugmentedState:
        ...

    def simulation_step(self, state: AugmentedState, action: int) -> SimulationResult:
        ...

    def reward(
        self, state: AugmentedState, action: int, next_state: AugmentedState
    ) -> float:
        ...

    def terminal(
        self, state: AugmentedState, action: int, next_state: AugmentedState
    ) -> bool:
        ...
