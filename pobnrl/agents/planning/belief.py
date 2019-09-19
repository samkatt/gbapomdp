""" contains algorithms and classes for maintaining beliefs """

from typing import Callable, Optional

from typing_extensions import Protocol
import numpy as np

from misc import POBNRLogger
from environments import Simulator

from .particle_filters import ParticleFilter, WeightedFilter, WeightedParticle


class BeliefUpdate(Protocol):
    """ Defines the signature of a update function for a particle """

    def __call__(
            self,
            belief: ParticleFilter,
            action: np.ndarray,
            observation: np.ndarray) -> ParticleFilter:
        """ function call signature for particle update functions

        Args:
             particle_filter: (`ParticleFilter`):
             action: (`np.ndarray`):
             observation: (`np.ndarray`):

        RETURNS (`ParticleFilter`):

        """


class BeliefManager(POBNRLogger):
    """ manages a belief """

    def __init__(
            self,
            reset_f: Callable[[], ParticleFilter],
            update_belief_f: BeliefUpdate,
            episode_reset_f: Optional[Callable[[ParticleFilter], ParticleFilter]] = None):
        """ Maintians a belief

        Manages belief by initializing, updating, and returning it.

        Args:
             reset_f: (`Callable[[], ParticleFilter]`): the function to call to reset the belief
             update_belief_f: (`BeliefUpdate`): the function to call to update the belief
             episode_reset_f: (`Optional[Callable[[` `ParticleFilter` `], `ParticleFilter` ]]`): the episode reset function to call

        Default value for `episode_reset_f` is to do the same as `reset_f`

        """

        POBNRLogger.__init__(self)

        self._reset = reset_f
        self._update = update_belief_f

        if episode_reset_f:
            self._episode_reset = episode_reset_f
        else:
            self._episode_reset = lambda _: self._reset()

        self._belief = self._reset()

    def reset(self) -> None:
        """ resets by sampling new belief """

        self._belief = self._reset()

        if self.log_is_on(POBNRLogger.LogLevel.V3):
            self.log(POBNRLogger.LogLevel.V3, f"Belief reset to {self._belief}")

    def episode_reset(self) -> None:
        """ resets the belief for a new episode """

        self._belief = self._episode_reset(self.particle_filter)

        if self.log_is_on(POBNRLogger.LogLevel.V3):
            self.log(POBNRLogger.LogLevel.V3, f"Belief reset for new episode {self._belief}")

    def update(self, action: int, observation: np.ndarray):
        """ updates belief given action and observation

        Args:
             action: (`int`):
             observation: (`np.ndarray`):

        """

        self._belief = self._update(belief=self._belief, action=action, observation=observation)

        if self.log_is_on(POBNRLogger.LogLevel.V3):
            self.log(POBNRLogger.LogLevel.V3, f"BELIEF: update after a({action}), o({observation}): {self._belief}")

    @property
    def particle_filter(self) -> ParticleFilter:
        """ returns the belief

        RETURNS (`ParticleFilter`):

        """
        return self._belief


def rejection_sampling(
        belief: ParticleFilter,
        action: np.ndarray,
        observation: np.ndarray,
        sim: Simulator) -> ParticleFilter:

    next_belief = type(belief)()

    while next_belief.size < belief.size:

        state = belief.sample()
        transition = sim.simulation_step(state, action)

        if np.all(transition.observation == observation):
            next_belief.add_particle(transition.state)

    return next_belief


def importance_sampling(
        belief: WeightedFilter,
        action: np.ndarray,
        observation: np.ndarray,
        sim: Simulator) -> WeightedFilter:

    next_belief = WeightedFilter()

    for _ in range(belief.size):

        # XXX: with resampling here
        state = belief.sample()

        transition = sim.simulation_step(state, action)
        weight = state.model.observation_prob(  # XXX assumes particles are AugmentedStates
            state.domain_state, action, transition.state.domain_state, observation
        )

        next_belief.add_weighted_particle(WeightedParticle(transition.state, weight))

    return next_belief
