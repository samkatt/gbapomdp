""" agents that act by learning a model of the environments """

from functools import partial
from typing import Optional

import numpy as np
from po_nrl.agents.agent import Agent
from po_nrl.agents.planning.belief import (BeliefManager,
                                           belief_update_factory,
                                           rejection_sampling)
from po_nrl.agents.planning.particle_filters import (FlatFilter,
                                                     ParticleFilter,
                                                     WeightedFilter)
from po_nrl.agents.planning.pouct import POUCT, RolloutPolicy
from po_nrl.analysis.augmented_beliefs import analyzer_factory
from po_nrl.domains import NeuralEnsemblePOMDP
from po_nrl.domains.gridverse_domain import GridverseDomain, rollout_policy
from po_nrl.environments import Environment, Simulator
from po_nrl.misc import POBNRLogger


class PrototypeAgent(Agent, POBNRLogger):
    """ default model-based agent """

    def __init__(self, planner: POUCT, belief_manager: BeliefManager):
        """ creates this agent with planner and belief manager

        Args:
             planner: (`po_nrl.agents.planning.pouct.POUCT`):
             BeliefManager: (`po_nrl.agents.planning.particle_filters.BeliefManager`):

        """

        POBNRLogger.__init__(self)

        self._planner = planner
        self._belief_manager = belief_manager

        self._last_action = -1

    def reset(self) -> None:
        """ resets belief """
        self._belief_manager.reset()

    def episode_reset(self, _observation: np.ndarray) -> None:
        """ resets belief

        Ignores observation for now

        Args:
             observation: (`np.ndarray`): the initial episode observation

        """

        self.log(POBNRLogger.LogLevel.V3, "Resetting agent for next episode")
        self._belief_manager.episode_reset()

    def select_action(self) -> int:
        """  runs PO-UCT on current belief

        RETURNS (`int`): action

        """

        self._last_action = self._planner.select_action(
            self._belief_manager.particle_filter
        )

        return self._last_action

    def update(self, observation: np.ndarray, _reward: float, _terminal: bool):
        """ calls at the end of a real step to allow the agent to update

        Will update the belief given the observation (and last action)

        Args:
             observation: (`np.ndarray`): the observation
             reward: (`float`): ignored
             terminal: (`bool`): ignored

        """

        self._belief_manager.update(self._last_action, observation)


def episode_reset_belief(
    p_filter: ParticleFilter, sim: NeuralEnsemblePOMDP
) -> ParticleFilter:
    """ resets the belief in between episodes

    In between episodes the (state) belief has to be reset to represent the
    initial POMDP state distribution. This is done here by resetting the POMDP
    state of each particle to some initial POMDP state.

    Args:
         p_filter: (`ParticleFilter`): belief to reset
         sim: (`NeuralEnsemblePOMDP`): the simulator used to sample states

    RETURNS (`ParticleFilter`):

    """

    for particle in p_filter:
        particle.domain_state = sim.sample_domain_start_state()

    return p_filter


def _create_learning_belief_manager(
    sim: NeuralEnsemblePOMDP, conf
) -> BeliefManager:
    """ returns a belief manager for the learning context

    return rejection sampling belief manager by setting the belief manager
    functionality correctly

    Args:
         sim: (`NeuralEnsemblePOMDP`): the simulator of the agent
         conf: program argument configurations

    RETURNS (`BeliefManager`): rejection sampling belief manager (for now)

    """

    assert conf.belief in [
        "rejection_sampling",
        "importance_sampling",
    ], f"belief {conf.belief} not legal"

    analyser = analyzer_factory(conf.domain, conf.domain_size)

    belief_update = belief_update_factory(conf, sim)

    episode_reset = partial(episode_reset_belief, sim=sim)

    if conf.belief == 'rejection_sampling':
        reset = partial(
            FlatFilter.create_from_process,
            sample_process=sim.sample_start_state,
            size=conf.num_particles,
        )
    elif conf.belief == 'importance_sampling':
        reset = partial(
            WeightedFilter.create_from_process,
            sample_process=sim.sample_start_state,
            size=conf.num_particles,
        )
    else:
        raise ValueError(
            f"belief must be 'rejection_sampling' or 'importance_sampling', not {conf.belief}"
        )

    return BeliefManager(
        reset_f=reset,
        update_belief_f=belief_update,
        episode_reset_f=episode_reset,
        belief_analyzer=analyser,
    )


def create_rollout_policy(domain: Environment) -> Optional[RolloutPolicy]:
    """returns, if available, a domain specific rollout policy

    Currently only returns for gridverse domain

    Args:
        domain (`Environment`): environment

    Returns:
        `Optional[RolloutPolicy]`:
    """

    if isinstance(domain, GridverseDomain):
        return partial(
            rollout_policy,
            encoding=domain._state_encoding,  # pylint: disable=protected-access
        )

    return None


def create_learning_agent(sim: NeuralEnsemblePOMDP, conf) -> PrototypeAgent:
    """ factory function to construct model based learning agents

    Args:
         sim: (`po_nrl.environments.Simulator`) simulator
         conf: (`namespace`) configurations

    RETURNS (`po_nrl.agents.model_based_agents.PrototypeAgent`)

    """

    pol = create_rollout_policy(conf.domain)
    planner = POUCT(
        sim,
        conf.num_sims,
        conf.exploration,
        conf.search_depth,
        conf.gamma,
        pol,
    )

    belief_manager = _create_learning_belief_manager(sim, conf)

    return PrototypeAgent(planner, belief_manager,)


def create_planning_agent(sim: Simulator, conf) -> PrototypeAgent:
    """ factory function to construct planning agents

    Args:
         sim: (`po_nrl.environments.Simulator`) simulator
         conf: (`namespace`) configurations

    RETURNS (`po_nrl.agents.model_based_agents.PrototypeAgent`)

    """

    if conf.belief != 'rejection_sampling':
        raise ValueError('belief must be rejection_sampling')

    pol = create_rollout_policy(conf.domain)
    planner = POUCT(
        sim,
        conf.num_sims,
        conf.exploration,
        conf.search_depth,
        conf.gamma,
        pol,
    )

    believe_manager = BeliefManager(
        reset_f=partial(
            FlatFilter.create_from_process,
            sample_process=sim.sample_start_state,
            size=conf.num_particles,
        ),
        update_belief_f=partial(rejection_sampling, sim=sim),
    )

    return PrototypeAgent(planner, believe_manager)
