""" agents that act by learning a model of the environments """

from collections import Counter
from functools import partial
import numpy as np

from agents.agent import Agent
from agents.planning.belief import BeliefManager, rejection_sampling, importance_sample_factory, noop_analysis
from agents.planning.particle_filters import ParticleFilter, FlatFilter, WeightedFilter
from agents.planning.pouct import POUCT
from analysis.augmented_beliefs import analyzer_factory
from domains import NeuralEnsemblePOMDP
from environments import Simulator
from misc import POBNRLogger


class PrototypeAgent(Agent, POBNRLogger):
    """ default model-based agent """

    def __init__(self, planner: POUCT, belief_manager: BeliefManager):
        """ creates this agent with planner and belief manager

        Args:
             planner: (`pobnrl.agents.planning.pouct.POUCT`):
             BeliefManager: (`pobnrl.agents.planning.particle_filters.BeliefManager`):

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

        self._last_action = self._planner.select_action(self._belief_manager.particle_filter)

        return self._last_action

    def update(
            self,
            observation: np.ndarray,
            _reward: float,
            _terminal: bool):
        """ calls at the end of a real step to allow the agent to update

        Will update the belief given the observation (and last action)

        Args:
             observation: (`np.ndarray`): the observation
             reward: (`float`): ignored
             terminal: (`bool`): ignored

        """

        self._belief_manager.update(self._last_action, observation)


def episode_reset_belief(
        p_filter: ParticleFilter,
        logger: POBNRLogger,
        sim: NeuralEnsemblePOMDP) -> ParticleFilter:
    """ resets the belief in between episodes

    In between episodes the (state) belief has to be reset to represent the
    initial POMDP state distribution. This is done here by resetting the POMDP
    state of each particle to some initial POMDP state.

    Args:
         p_filter: (`ParticleFilter`): belief to reset
         logger: (`POBNRLogger`): used for logging diagnostics
         sim: (`NeuralEnsemblePOMDP`): the simulator used to sample states

    RETURNS (`ParticleFilter`):

    """

    for particle in p_filter:
        particle.domain_state = sim.sample_domain_start_state()

    # TODO: abstract this away somehow
    if logger.log_is_on(POBNRLogger.LogLevel.V3):

        logger.log(
            logger.LogLevel.V3,
            f'Model density:'
            f'{sorted(Counter([particle.model for particle in p_filter]).items(), key=lambda x: id(x[0]))}'
        )
        logger.log(
            logger.LogLevel.V3,
            f'States after reset {str(Counter([x.domain_state[0] for x in p_filter]))}'
        )

    return p_filter


def _create_learning_belief_manager(sim: NeuralEnsemblePOMDP, conf) -> BeliefManager:
    """ returns a belief manager for the learning context

    return rejection sampling belief manager by setting the belief manager
    functionality correctly

    Args:
         sim: (`NeuralEnsemblePOMDP`): the simulator of the agent
         conf: program argument configurations

    RETURNS (`BeliefManager`): rejection sampling belief manager (for now)

    """

    if conf.analyse_belief:
        analyser = analyzer_factory(conf.domain)
    else:
        analyser = noop_analysis

    if conf.belief == 'rejection_sampling':

        # return a rejection sampling belief manager:
        return BeliefManager(
            reset_f=partial(
                FlatFilter.create_from_process,
                sample_process=sim.sample_start_state,
                size=conf.num_particles
            ),
            update_belief_f=partial(rejection_sampling, sim=sim),
            episode_reset_f=partial(
                episode_reset_belief,
                logger=POBNRLogger('BeliefManager'),
                sim=sim
            ),
            belief_analyzer=analyser
        )

    if conf.belief == 'importance_sampling':

        return BeliefManager(
            reset_f=partial(
                WeightedFilter.create_from_process,
                sample_process=sim.sample_start_state,
                size=conf.num_particles
            ),
            update_belief_f=importance_sample_factory(
                conf.perturb_stdev, conf.backprop
            ),
            episode_reset_f=partial(
                episode_reset_belief,
                logger=POBNRLogger('BeliefManager'),
                sim=sim
            ),
            belief_analyzer=analyser
        )

    raise ValueError(f"belief must be 'rejection_sampling' or 'importance_sampling', not {conf.belief}")


def create_learning_agent(sim: NeuralEnsemblePOMDP, conf) -> PrototypeAgent:
    """ factory function to construct model based learning agents

    Args:
         sim: (`pobnrl.environments.Simulator`) simulator
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.model_based_agents.PrototypeAgent`)

    """

    planner = POUCT(
        sim,
        conf.num_sims,
        conf.exploration,
        conf.horizon,
        conf.gamma
    )

    belief_manager = _create_learning_belief_manager(sim, conf)

    return PrototypeAgent(
        planner,
        belief_manager,
    )


def create_planning_agent(sim: Simulator, conf) -> PrototypeAgent:
    """ factory function to construct planning agents

    Args:
         sim: (`pobnrl.environments.Simulator`) simulator
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.model_based_agents.PrototypeAgent`)

    """

    if conf.belief != 'rejection_sampling':
        raise ValueError('belief must be rejection_sampling')

    planner = POUCT(
        sim,
        conf.num_sims,
        conf.exploration,
        conf.horizon,
        conf.gamma
    )

    believe_manager = BeliefManager(
        reset_f=partial(
            FlatFilter.create_from_process,
            sample_process=sim.sample_start_state, size=conf.num_particles
        ),
        update_belief_f=partial(rejection_sampling, sim=sim)
    )

    return PrototypeAgent(
        planner,
        believe_manager
    )
