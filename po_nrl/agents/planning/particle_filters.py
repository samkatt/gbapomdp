""" particle filters represent beliefs over the state space as a bag of particles

Contains:
    * particle filters
    * update functions (rejection sampling, importance sampling)
    * belief managers
"""

from typing import Any, Callable, List, Iterator
import abc
import random

import numpy as np


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

    @classmethod
    def create_from_process(
            cls,
            sample_process: Callable[[], Any],
            size: int) -> 'ParticleFilter':
        """ create a flat filter through some sampling process

        Args:
             sample_process: (`Callable[[], Any]`): function to generate samples
             size: (`int`): number of particles

        RETURNS (`FlatFilter`):

        """

        new_filter = cls()

        for _ in range(size):
            new_filter.add_particle(sample_process())

        return new_filter


class FlatFilter(ParticleFilter):
    """ a filter where particles have no weights """

    def __init__(self) -> None:
        """ create a flat filter """

        self._particles: List[Any] = []

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
        return f"FlatFilter of size {self.size}"


class WeightedParticle():
    """ a particle associated with a weight """

    def __init__(self, value: Any, weight: float = 1):
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

    def __init__(self) -> None:
        """ create a weighted filter """

        self._total_weight = .0
        self._particles: List[WeightedParticle] = []

    def add_particle(self, particle: Any) -> None:
        """ adds a particle to the filter

        Args:
             particle: (`Any`): particle to be added
             weight: (`float`): weight of particle to be added

        RETURNS (`None`):

        """

        self._total_weight += 1.
        self._particles.append(WeightedParticle(particle, 1.))

    def add_weighted_particle(self, particle: WeightedParticle) -> None:
        """ adds a `WeightedParticle` to the filter

        Args:
             particle: (`WeightedParticle`): to particle to be added

        RETURNS (`None`):

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

        acc = .0
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
    def particles(self) -> Iterator[WeightedParticle]:
        """ returns an iterator over the weighted particles

        Args:

        RETURNS (`iter`):

        """
        return iter(self._particles)

    @property
    def size(self) -> int:
        """ returns size of particle fitler (number of samples)

        RETURNS (`int`):
        """
        return len(self._particles)

    def __iter__(self):
        """ iterate over samples in filter """
        return (p.value for p in self._particles)

    def __repr__(self) -> str:
        return f'WeightedFilter of {self.size} particles and {self._total_weight} total weight'

    # TODO: test and doc
    def effective_sample_size(self) -> float:
        return 1 / np.sum([pow(p.weight / self._total_weight, 2) for p in self.particles])


# TODO: test and doc
def resample(belief: WeightedFilter) -> WeightedFilter:

    next_belief = WeightedFilter()

    for _ in range(belief.size):

        next_belief.add_particle(belief.sample())

    return next_belief
