""" particle filters represent beliefs over the state space as a bag of particles

Contains:
    * particle filters
    * update functions (rejection sampling, importance sampling)
    * belief managers
"""

from typing import Any, Callable, List, Optional
import abc
import random

from typing_extensions import Protocol
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

    @staticmethod
    def resample(p_filter: 'ParticleFilter', constructor: Callable[[], 'ParticleFilter'], num: int = 0) -> 'ParticleFilter':
        """ performs a resample

        TODO: remove `constructor` and replace construction with type(p_filter)()

        Args:
             p_filter: (`ParticleFilter`): the filter to resample from
             constructor: (`Callable[[], ` `ParticleFilter` ` ]`): constructor of new particle filter
             num: (`int`): set to `p_filter` size if not provided

        RETURNS (`ParticleFilter`):

        """

        assert num >= 0, f"cannot resample less than 0 ({num}) samples"

        if num == 0:
            num = p_filter.size

        new_filter = constructor()

        for _ in range(num):
            new_filter.add_particle(p_filter.sample().particle)

        return new_filter


class FlatFilter(ParticleFilter):
    """ a filter where particles have no weights """

    def __init__(self):
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

    @classmethod
    def create_from_process(cls, sample_process: Callable[[], Any], size: int) -> 'FlatFilter':
        """ create a flat filter through some sampling process

        Args:
             sample_process: (`Callable[[], Any]`): function to generate samples
             size: (`int`): number of particles

        RETURNS (`FlatFilter`):

        """
        flatfilter = cls()

        for _ in range(size):
            flatfilter.add_particle(sample_process())

        return flatfilter


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

    def __init__(self):
        """ create a weighted filter """

        self._total_weight = .0
        self._particles: List[WeightedParticle] = []

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


class BeliefUpdate(Protocol):
    """ Defines the signature of a update function for a particle """

    def __call__(
            self,
            particle_filter: ParticleFilter,
            action: int,
            observation: int) -> ParticleFilter:
        """ function call signature for particle update functions

        Args:
             particle_filter: (`ParticleFilter`):
             action: (`int`):
             observation: (`int`):

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

        self._belief = self._update(particle_filter=self._belief, action=action, observation=observation)

        if self.log_is_on(POBNRLogger.LogLevel.V4):
            self.log(POBNRLogger.LogLevel.V4, f"BELIEF: update after a({action}), o({observation}): {self._belief}")

    @property
    def particle_filter(self) -> ParticleFilter:
        """ returns the belief

        RETURNS (`ParticleFilter`):

        """
        return self._belief
