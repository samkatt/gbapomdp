""" particle filters represent beliefs over the state space as a bag of particles

Contains:
    * particle filters
    * update functions (rejection sampling, importance sampling)
    * belief managers
"""

import abc
from copy import deepcopy
from typing import Any, Callable, Iterable, Tuple

import numpy as np
import pomdp_belief_tracking.pf.importance_sampling as LibIS
import pomdp_belief_tracking.pf.particle_filter as LibPF
import pomdp_belief_tracking.pf.rejection_sampling as LibRS
from general_bayes_adaptive_pomdps.domains.learned_environments import BADDr
from general_bayes_adaptive_pomdps.environments import Simulator


class Belief(abc.ABC):
    """ a distribution defined by a bag of particles """

    def __call__(self) -> Any:
        """Belief base requirement is to be able to sample

        Returns:
            Any: a particle
        """

    @abc.abstractmethod
    def apply(self, f: Callable[[Any], Any]) -> None:
        """Apply ``f`` to the belief

        Will affect the complete belief

        Args:
            f (Callable[[Any], Any]): function to apply to ``self``

        Returns:
            None
        """


class InitialStateSampler(Belief):
    """A :class:`Belief` that samples initial states"""

    def __init__(self, sampler: Callable[[], np.ndarray]):
        """Wraps ``sampler``

        Implements :meth:`__call__` by calling ``sampler``

        XXX: :meth:`apply` does **nothing**

        Args:
            sampler (Callable[[], np.ndarray]): a method to sample states from
        """
        self.sampler = sampler

    def __call__(self) -> np.ndarray:
        """Samples an initial state from ``sampler`` given at initiation"""
        return self.sampler()

    def apply(self, f: Callable[[Any], Any]) -> None:
        """You should not be able to do this"""


class ParticleFilter(Belief):
    """A :class:`Belief` approximation through particles"""

    def __init__(self, particles: Iterable[Any]):
        """Creates a PF from ``particles``

        Stores a list of ``particles``

        Args:
            particles (Iterable[Any]): samples to store
        """
        self.particle_filter = LibPF.ParticleFilter(list(particles))

    def __call__(self) -> Any:
        """Samples a particle

        Returns:
            Any: probably state
        """
        return self.particle_filter()

    def apply(self, f: Callable[[Any], Any]):
        """Applies ``f`` to all particles

        Args:
            f (Callable[[Any], Any])
        """

        def apply_f(p):
            return LibPF.Particle(f(p.state), p.weight)

        new_particles = list(map(apply_f, self.particle_filter.particles))
        self.particle_filter = LibPF.ParticleFilter.from_particles(new_particles)


def create_importance_sampling(sim: BADDr, num_samples: int):
    """Creates importance sampling :class:`general_bayes_adaptive_pomdps.agents.planning.belief.BeliefUpdate`

    Given the ``sim``, used to simulate steps and weight observation
    probabilities, create a method that does importance sampling as belief
    update on :class:`ParticleFilter`.

    XXX: does not implement re-sampling

    Args:
        sim (BADDr): the GBA-POMDP, used to simulate and weight steps
        num_samples (int): number of samples to populate :class:`Belief` with

    Returns:
        general_bayes_adaptive_pomdps.agents.planning.belief.BeliefUpdate: a way of updating the belief
    """

    def belief_update(belief: Belief, action: int, observation: np.ndarray) -> Belief:
        """The actual belief update to be returned

        XXX: does not implement re-sampling

        Args:
            belief (Belief): belief at time step t
            action (int): action at time step t
            observation (np.ndarray): observation at time step t+1

        Returns:
            Belief: belief at time step t+1
        """

        # two possibilities:
        if isinstance(belief, ParticleFilter):
            # 1: incoming belief is a :class:`ParticleFilter`
            particles = belief.particle_filter.particles
        else:
            # 2: incoming belief is a generative function of initial states

            # create weighted list of particles
            assert num_samples > 0
            particles = ParticleFilter(
                belief() for _ in range(num_samples)
            ).particle_filter.particles

        def proposal(
            s: BADDr.AugmentedState, _: Any
        ) -> Tuple[BADDr.AugmentedState, Any]:
            return sim.simulation_step(s, action).state, {"state": s}

        def weighting(
            proposal: BADDr.AugmentedState,
            sample_ctx: Any,
            _: Any,
        ) -> float:
            o_model = proposal.model.observation_model(
                sample_ctx["state"].domain_state, action, proposal.domain_state
            )
            return np.prod(
                [distr[feature] for distr, feature in zip(o_model, observation)]
            )

        pf, _ = LibIS.general_importance_sample(proposal, weighting, particles)
        assert isinstance(pf, LibPF.ParticleFilter)

        next_b = ParticleFilter([])
        next_b.particle_filter = pf

        return next_b

    return belief_update


def create_rejection_sampling(sim: Simulator, num_samples: int):
    """Creates rejection sampling :class:`general_bayes_adaptive_pomdps.agents.planning.belief.BeliefUpdate`

    Given the ``sim``, used to simulate steps, create a method that does
    rejection sampling as belief update on :class:`ParticleFilter`.

    Args:
        sim (Simulator): used to simulate steps
        num_samples (int): number of samples to populate :class:`Belief` with

    Returns:
        general_bayes_adaptive_pomdps.agents.planning.belief.BeliefUpdate: a way of updating the belief
    """

    if isinstance(sim, BADDr):
        step = sim.simulation_step_without_updating_theta

        def process_acpt(ss, ctx, _):
            # update the parameters of the augmented state
            copy = deepcopy(ss)
            sim.update_theta(
                copy.model,
                ctx["state"].domain_state,
                ctx["action"],
                copy.domain_state,
                ctx["observation"],
            )
            return copy

    else:
        # regular POMDP, not BADDr
        step = sim.simulation_step
        process_acpt = LibRS.accept_noop

    def belief_sim(s: np.ndarray, a: int) -> Tuple[np.ndarray, np.ndarray]:
        out = step(s, a)
        return out.state, out.observation

    def belief_update(
        belief: Belief, action: np.ndarray, observation: np.ndarray
    ) -> Belief:
        pf, _ = LibRS.rejection_sample(
            sim=belief_sim,  # type: ignore
            observation_matches=np.array_equal,
            n=num_samples,
            process_acpt=process_acpt,
            process_rej=LibRS.reject_noop,
            initial_state_distribution=belief,
            a=action,
            o=observation,
        )

        assert isinstance(pf, LibPF.ParticleFilter)

        next_b = ParticleFilter([])
        next_b.particle_filter = pf

        return next_b

    return belief_update


def create_belief_update(conf, sim: Simulator):
    """Factory function for :class:`general_bayes_adaptive_pomdps.agents.planning.belief.BeliefUpdate`

    Uses the configurations to decided what type of belief update to construct.
    The simulator ``sim`` is then used to actually implement it.

    Args:
        conf: arg-parser name space with configurations
        sim (Simulator): POMDP simulator

    Returns:
        general_bayes_adaptive_pomdps.agents.planning.belief.BeliefUpdate

    """

    if conf.belief == "rejection_sampling":
        return create_rejection_sampling(sim, conf.num_particles)
    if conf.belief == "importance_sampling":
        assert isinstance(sim, BADDr)
        return create_importance_sampling(sim, conf.num_particles)

    assert False, f"{conf.belief} is not a valid belief configuration"
