""" agents that act by learning a model of the environments """

from functools import partial

import numpy as np

from general_bayes_adaptive_pomdps.agents.agent import Agent
from general_bayes_adaptive_pomdps.agents.planning.belief import (
    BeliefManager,
    create_beliefupdate_for_learning,
)
from general_bayes_adaptive_pomdps.agents.planning.particle_filters import (
    Belief,
    InitialStateSampler,
    create_belief_update,
)
from general_bayes_adaptive_pomdps.agents.planning.pouct import (
    Planner,
    RolloutPolicy,
    create_planner,
    random_policy,
)
from general_bayes_adaptive_pomdps.analysis.augmented_beliefs import analyzer_factory
from general_bayes_adaptive_pomdps.domains.gridverse_domain import GridverseDomain
from general_bayes_adaptive_pomdps.domains.gridverse_domain import (
    default_rollout_policy as gridverse_regular_rollout,
)
from general_bayes_adaptive_pomdps.domains.gridverse_domain import (
    straight_or_turn_policy,
)
from general_bayes_adaptive_pomdps.environments import Simulator
from general_bayes_adaptive_pomdps.misc import POBNRLogger
from general_bayes_adaptive_pomdps.models.baddr import BADDr


class PrototypeAgent(Agent, POBNRLogger):
    """ default model-based agent """

    def __init__(self, planner: Planner, belief_manager: BeliefManager):
        """creates this agent with planner and belief manager

        Args:
             planner: (`general_bayes_adaptive_pomdps.agents.planning.pouct.Planner`):
             BeliefManager: (`general_bayes_adaptive_pomdps.agents.planning.particle_filters.BeliefManager`):
        """

        POBNRLogger.__init__(self)

        self._planner = planner
        self._belief_manager = belief_manager

        self._last_action = -1

    def reset(self) -> None:
        """ resets belief """
        self._belief_manager.reset()

    def episode_reset(self, _observation: np.ndarray) -> None:
        """resets belief

        Ignores observation for now

        Args:
             observation: (`np.ndarray`): the initial episode observation

        """

        self.log(POBNRLogger.LogLevel.V3, "Resetting agent for next episode")
        self._belief_manager.episode_reset()

    def select_action(self) -> int:
        """runs PO-UCT on current belief

        RETURNS (`int`): action

        """

        self._last_action = self._planner(self._belief_manager.particle_filter)

        return self._last_action

    def update(self, observation: np.ndarray, _reward: float, _terminal: bool):
        """calls at the end of a real step to allow the agent to update

        Will update the belief given the observation (and last action)

        Args:
             observation: (`np.ndarray`): the observation
             reward: (`float`): ignored
             terminal: (`bool`): ignored

        """

        self._belief_manager.update(self._last_action, observation)


def episode_reset_belief(p_filter: Belief, sim: BADDr) -> Belief:
    """resets the belief in between episodes

    In between episodes the (state) belief has to be reset to represent the
    initial POMDP state distribution. This is done here by resetting the POMDP
    state of each particle to some initial POMDP state.

    Args:
         p_filter: (`Belief`): belief to reset
         sim: (`BADDr`): the simulator used to sample states

    RETURNS (`Belief`):

    """

    def reset_domain_state(
        s: BADDr.AugmentedState,
    ) -> BADDr.AugmentedState:
        s.domain_state = sim.sample_domain_start_state()
        return s

    p_filter.apply(reset_domain_state)

    return p_filter


def _create_learning_belief_manager(sim: BADDr, conf) -> BeliefManager:
    """returns a belief manager for the learning context

    return rejection sampling belief manager by setting the belief manager
    functionality correctly

    Args:
         sim: (`BADDr`): the simulator of the agent
         conf: program argument configurations

    RETURNS (`BeliefManager`): rejection sampling belief manager (for now)

    """

    assert conf.belief in [
        "rejection_sampling",
        "importance_sampling",
    ], f"belief {conf.belief} not legal"

    analyser = analyzer_factory(conf.domain, conf.domain_size)

    belief_update = create_beliefupdate_for_learning(conf, sim)

    episode_reset = partial(episode_reset_belief, sim=sim)

    def reset():
        return InitialStateSampler(sim.sample_start_state)

    return BeliefManager(
        reset_f=reset,
        update_belief_f=belief_update,
        episode_reset_f=episode_reset,
        belief_analyzer=analyser,
    )


def create_rollout_policy(domain: Simulator, rollout_descr: str) -> RolloutPolicy:
    """returns, if available, a domain specific rollout policy

    Currently only returns for gridverse domain

    Args:
        domain (`Simulator`): environment
        rollout_descr (`str`):

    Returns:
        `Optional[RolloutPolicy]`:
    """

    if isinstance(domain, GridverseDomain):
        if rollout_descr == "default":
            return partial(
                gridverse_regular_rollout,
                encoding=domain._state_encoding,  # pylint: disable=protected-access
            )
        if rollout_descr == "gridverse-extra":
            return partial(
                straight_or_turn_policy,
                encoding=domain._state_encoding,  # pylint: disable=protected-access
            )

    if rollout_descr:
        raise ValueError(
            f"{rollout_descr} not accepted as rollout policy for domain {domain}"
        )

    return partial(random_policy, action_space=domain.action_space)


def create_learning_agent(sim: BADDr, conf, domain: Simulator) -> PrototypeAgent:
    """factory function to construct model based learning agents

    NOTE: unfortunately requires `domain` in order to pass it onto
    `create_rollout_policy`. Ideally the learning agent does not know anything
    about the environment, but system design is hard, so here we are.

    Args:
         sim: (`general_bayes_adaptive_pomdps.environments.Simulator`) simulator
         conf: (`namespace`) configurations
         domain: (`general_bayes_adaptive_pomdps.environments.Simulator`) for `create_rollout_policy`

    RETURNS (`general_bayes_adaptive_pomdps.agents.model_based_agents.PrototypeAgent`)

    """

    pol = create_rollout_policy(domain, conf.rollout_policy)

    pol_descr = str(pol)

    def rollout(augmented_state) -> int:
        """
        So normally PO-UCT expects states to be numpy arrays and everything is
        dandy, but we are planning in augmented space here in secret. So the
        typical rollout policy of the environment will not work: it does not
        expect an `AugmentedState`. So here we gently provide it the underlying
        state and all is well
        """
        return pol(augmented_state.domain_state)

    # Turns out that the most ugly code here really exists for some stupid
    # logging. I dislike how there is no way of telling what function `rollout`
    # calls, so here I am inserting that information. Any normal programmer
    # would have been able to think of a better way, but alas there is only me,
    # so here we are
    rollout.__str__ = lambda: pol_descr  # type: ignore

    planner = create_planner(
        sim,
        rollout,  # type: ignore
        conf.num_sims,
        conf.exploration,
        conf.search_depth,
        conf.gamma,
    )

    belief_manager = _create_learning_belief_manager(sim, conf)

    return PrototypeAgent(
        planner,
        belief_manager,
    )


def create_planning_agent(sim: Simulator, conf) -> PrototypeAgent:
    """factory function to construct planning agents

    Args:
         sim: (`general_bayes_adaptive_pomdps.environments.Simulator`) simulator
         conf: (`namespace`) configurations

    RETURNS (`general_bayes_adaptive_pomdps.agents.model_based_agents.PrototypeAgent`)

    """

    if conf.belief != "rejection_sampling":
        raise ValueError("belief must be rejection_sampling")

    pol = create_rollout_policy(sim, conf.rollout_policy)
    planner = create_planner(
        sim,
        pol,
        conf.num_sims,
        conf.exploration,
        conf.search_depth,
        conf.gamma,
    )

    def reset_f():
        return InitialStateSampler(sim.sample_start_state)

    update_belief_f = create_belief_update(conf, sim)

    believe_manager = BeliefManager(
        reset_f,
        update_belief_f,
    )

    return PrototypeAgent(planner, believe_manager)
