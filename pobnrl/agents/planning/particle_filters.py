""" beliefs are distributions over states

Contains:
    * beliefs (particle filters)
    * update functions (rejection sampling, importance sampling)
    * belief managers
"""

from collections import Counter
from typing import Any, Callable
import abc
import random

import numpy as np

from misc import POBNRLogger


class ParticleFilter(abc.ABC):
    """ a distribution defined by a bag of particles """

    @abc.abstractmethod
    def add_particle(self, particle: Any):
        """ adds a particle to the filter

        Args:
             particle: (`Any`):

        """

    @abc.abstractmethod
    def sample(self) -> Any:
        """ randomly returns a particle

        RETURNS (`Any`): particle

        """

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """ returns size of particle fitler (number of samples)

        RETURNS (`int`):
        """

    @abc.abstractmethod
    def __iter__(self):
        """ returns an iterator """


class FlatFilter(ParticleFilter):
    """ a filter where particles have no weights """

    def __init__(self):
        """ create a flat filter """

        self._particles = []

    def add_particle(self, particle: Any):
        """ adds a particle to the filter

        Args:
             particle: (`Any`):

        """
        self._particles.append(particle)

    def sample(self) -> Any:
        """ randomly returns a particle

        RETURNS (`Any`): particle

        """
        return random.choice(self._particles)

    @property
    def size(self) -> int:
        """ returns size of particle fitler (number of samples)

        RETURNS (`int`):
        """
        return len(self._particles)

    def __iter__(self):
        return iter(self._particles)

    def __repr__(self) -> str:
        try:
            return f"FlatFilter with: {Counter(self._particles)}"
        except TypeError:
            return f"FlatFilter of size {self.size}"


class WeightedParticle():
    """ a particle associated with a weight """

    def __init__(self, value: Any, weight: float):
        """ creates a particle `value` of weight `weight`

        Args:
             value: (`Any`): whatever the particle contains
             weight: (`float`): the weight/probability of the particle

        """

        if weight < 0:
            raise ValueError("weights should be positive")

        self._weight = weight
        self._val = value

    @property
    def weight(self) -> float:
        """ returns the weight of the particle

        RETURNS (`float`):

        """
        return self._weight

    @weight.setter
    def weight(self, weight: float):
        """ sets weight of particles

        Args:
             weight: (`float`): must be > 0

        """

        assert weight > 0
        self._weight = weight

    @property
    def value(self) -> Any:
        """ returns the value of the particle

        RETURNS (`Any`):

        """
        return self._val

    @value.setter
    def value(self, value: Any):
        """ sets value of particle

        Args:
             value: (`Any`):

        """

        self._val = value


class WeightedFilter(ParticleFilter):
    """ a filter where particles are associated with a weight """

    def __init__(self):
        """ create a weighted filter """

        self._total_weight = 0
        self._particles = []

    def add_particle(self, particle: WeightedParticle):
        """ adds a (weightedfilter.)particle to the filter

        Args:
             particle: (`WeightedParticle`): weighted particle to be added

        """
        if particle.weight < 0:
            raise ValueError("weights should be positive")

        self._total_weight += particle.weight
        self._particles.append(particle)

    def sample(self) -> Any:
        """ randomly returns a particle

        RETURNS (`Any`): particle

        """

        rand = random.uniform(0, self._total_weight)

        acc = 0
        # loop through particles until we accumulate more weight than sampled
        for particle in self._particles:
            acc += particle.weight

            if acc > rand:
                return particle.value

        raise IndexError(
            f"sampled weight {rand} (< stored total weight",
            f"{self._total_weight}) > real total weight {acc}"
        )

    @property
    def size(self) -> int:
        """ returns size of particle fitler (number of samples)

        RETURNS (`int`):
        """
        return len(self._particles)

    def __iter__(self):
        """ iterate over samples in filter """
        return iter(self._particles)


def rejection_sampling(
        particle_filter: ParticleFilter,
        process_sample_f: Callable[[Any], Any],
        accept_f: Callable[[Any], bool],
        extract_particle_f: Callable[[Any], Any] = lambda x: x) -> ParticleFilter:
    """ applies rejection sampling on particle_filter

    Will sample, process, accept and then extract particles from
    `particle_filter` untill a particle filter of accepted updated samples with
    the same size is generated

    Args:
         particle_filter: (`ParticleFilter`):
         process_sample_f: (`Callable[[Any], Any]`): processes samples in `particle_filter`
         accept_f: (`Callable[[Any], bool]`): tests processed sample
         extract_particle_f: (`Callable[[Any], Any]`): takes new particle from processed sample (default is identity)

    RETURNS (`ParticleFilter`): particle filter with accepted, updated particles

    """

    new_pf = type(particle_filter)()

    while new_pf.size < particle_filter.size:

        sample = particle_filter.sample()
        result = process_sample_f(sample)

        if accept_f(result):
            new_pf.add_particle(extract_particle_f(result))

    return new_pf


def importance_sampling(
        particle_filter: WeightedFilter,
        process_sample_f: Callable[[Any], Any],
        weight_f: Callable[[Any], float],
        extract_particle_f: Callable[[Any], Any] = lambda x: x) -> WeightedFilter:
    """ returns a updated weighted filter according to provided functions

    Will sample, weight, and extract particles from `particle_filter` untill a
    particle filter of accepted updated samples with the same size is generated

    Args:
         particle_filter: (`WeightedFilter`):
         process_sample_f: (`Callable[[Any], Any]`): processes samples in `particle_filter`
         weight_f: (`Callable[[Any], float]`): weights the processed sample
         extract_particle_f: (`Callable[[Any], Any]`): takes new particle from processed sample (default is identity)

    RETURNS (`WeightedFilter`): particle fitler with weighted, updated particles

    """

    for sample in particle_filter:

        result = process_sample_f(sample.value)

        sample.weight *= weight_f(result)
        sample.value = extract_particle_f(result)

    return particle_filter


class BeliefManager(POBNRLogger):
    """ manages a belief """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            num_particles: int,
            filter_type: Callable[[], ParticleFilter],
            sample_particle_f: Callable[[], Any],
            update_belief_f: Callable[[ParticleFilter, int, int], ParticleFilter],
            reset_particle_f: Callable[[Any], Any] = None):
        """ Maintians a belief

        Manages belief by initializing, updating, and returning it.

        Args:
             num_particles: (`int`): number of particles to hold
             filter_type: (`Callable[[],` `ParticleFilter` `],`) particle filter constructor
             sample_particle_f: (`Callable[[], Any]`): function that samples particles
             update_belief_f: (`Callable[[` `ParticleFilter`, `int, int],` `ParticleFilter` `]`): how to update the belief
             reset_particle_f: ('Callable[[Any], Any]'): how to reset a particle to start state (defaults to `sample_particle_f`)
        """

        POBNRLogger.__init__(self)

        self._size = num_particles
        self._filter_type = filter_type

        self._sample_particle_f = sample_particle_f
        self._update = update_belief_f
        self._reset_particle_f = reset_particle_f if reset_particle_f is not None else lambda _: self._sample_particle_f()

        self._belief = self._filter_type()

    def reset(self):
        """ resets by sampling new belief """

        self._belief = self._filter_type()

        # simply sample new states
        for _ in range(self._size):
            self._belief.add_particle(self._sample_particle_f())

        if self.log_is_on(POBNRLogger.LogLevel.V2):
            self.log(POBNRLogger.LogLevel.V2, f"Belief reset to {self._belief}")

    def episode_reset(self):
        """ resets the belief for a new episode """

        for particle in self._belief:
            self._reset_particle_f(particle)

        if self.log_is_on(POBNRLogger.LogLevel.V2):
            self.log(POBNRLogger.LogLevel.V2, f"Belief reset for new episode {self._belief}")

    def update(self, action: int, observation: np.ndarray):
        """ updates belief given action and observation

        Args:
             action: (`int`):
             observation: (`np.ndarray`):

        """

        self._belief = self._update(self._belief, action, observation)

        if self.log_is_on(POBNRLogger.LogLevel.V3):
            self.log(POBNRLogger.LogLevel.V3, f"BELIEF: update after a({action}), o({observation}): {self._belief}")

    @property
    def particle_filter(self) -> ParticleFilter:
        """ returns the belief

        RETURNS (`ParticleFilter`):

        """
        return self._belief
