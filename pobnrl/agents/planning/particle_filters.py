""" beliefs are distributions over states

Contains
* flat filter
* weighted filter

"""

from collections import Counter
from typing import Any, Callable
import abc
import random

import numpy as np

from misc import POBNRLogger, LogLevel


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

        Args:

        RETURNS (`Any`): particle

        """

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """ returns size of particle fitler (number of samples)

        RETURNS (`int`):
        """


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

        Args:

        RETURNS (`Any`): particle

        """
        return random.choice(self._particles)

    @property
    def size(self) -> int:
        """ returns size of particle fitler (number of samples)

        RETURNS (`int`):
        """
        return len(self._particles)

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

        Args:

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

        Args:

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

        Args:

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

    def __init__(
            self,
            sample_particle_f: Callable[[], Any],
            belief_type,  # class of particle filter to use
            update_belief_f: Callable[[ParticleFilter, int, int], ParticleFilter],
            num_particles):
        """ Maintians a belief

        Manages belief by initializing, updating, and returning it.

        Assumes the belief is over POMDP states.

        Args:
             sample_particle_f: (`Callable[[], Any]`): function that samples particles
             belief_type: (`class`) particle filterconstructor()
             update_belief_f: (`Callable[[` `ParticleFilter`, `int, int],` `ParticleFilter` `]`): how to update the belief
             num_particles: (`int`): number of particles to hold

        """

        POBNRLogger.__init__(self)

        self._size = num_particles
        self._belief_type = belief_type

        self._sample_particle_f = sample_particle_f
        self._update = update_belief_f

        self._belief = self._belief_type()

    def reset(self):
        """ resets by sampling new belief """

        self._belief = self._belief_type()

        # simply sample new states
        for _ in range(self._size):
            self._belief.add_particle(self._sample_particle_f())

        if self.log_is_on(LogLevel.V2):
            self.log(LogLevel.V2, f"Belief reset to {self._belief}")

    def update(self, action: int, observation: np.ndarray):
        """ updates belief given action and observation

        Args:
             action: (`int`):
             observation: (`np.ndarray`):

        """

        self._belief = self._update(self._belief, action, observation)

        if self.log_is_on(LogLevel.V3):
            self.log(LogLevel.V3, f"BELIEF: update after a({action}), o({observation}): {self._belief}")

    @property
    def belief(self) -> ParticleFilter:
        """ returns the belief

        TODO: rename to particle_filter

        Args:

        RETURNS (`ParticleFilter`):

        """
        return self._belief
