""" agents that act by learning a model of theenvironments """

from functools import partial
import numpy as np

from domains import NeuralEnsemblePOMDP
from environments import Simulator
from misc import POBNRLogger

from .agent import Agent
from .planning.particle_filters import BeliefManager, rejection_sampling
from .planning.particle_filters import ParticleFilter, FlatFilter
from .planning.pouct import POUCT


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

    @property
    def current_belief(self) -> ParticleFilter:
        """ get the current belief of the agent

        RETURNS (`ParticleFilter`):

        """
        return self._belief_manager.particle_filter

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


def belief_rejection_sampling(
        sim: Simulator,
        particle_filter: ParticleFilter,
        action: int,
        observation: np.ndarray) -> ParticleFilter:
    """ performs rejection sampling on the particle_filter given action and observation

    Args:
         sim: the simulator used to simulate steps
         particle_filter: (`ParticleFilter`): belief at t
         action: (`int`): chosen action
         observation: (`np.ndarray`): perceived observation

    RETURNS (`ParticleFilter`): belief at t+1

    """

    return rejection_sampling(
        particle_filter,
        process_sample_f=partial(sim.simulation_step, action=action),
        accept_f=lambda interaction: np.all(interaction.observation == observation),
        extract_particle_f=lambda interaction: interaction.state,
    )


def _create_learning_belief_manager(sim: NeuralEnsemblePOMDP, conf) -> BeliefManager:
    """ returns a belief manager for the learning context

    return rejection sampling belief manager by setting the belief manager
    functionality correctly

    Args:
         sim: (`NeuralEnsemblePOMDP`): the simulator of the agent
         conf: program argument configurations

    RETURNS (`BeliefManager`): rejection sampling belief manager (for now)

    """

    if conf.belief != 'rejection_sampling':
        raise ValueError(f'belief must be rejection_sampling, not {conf.belief}')

    def reset_particle(augmented_state: NeuralEnsemblePOMDP.AugmentedState):
        augmented_state.domain_state = sim.sample_domain_start_state()

    # return belief manager with rejection sampling functionality as defined above
    return BeliefManager(
        num_particles=conf.num_particles,
        filter_type=FlatFilter,
        sample_particle_f=sim.sample_start_state,
        update_belief_f=partial(belief_rejection_sampling, sim=sim),
        reset_particle_f=reset_particle
    )


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
        num_particles=conf.num_particles,
        filter_type=FlatFilter,
        sample_particle_f=sim.sample_start_state,
        update_belief_f=partial(belief_rejection_sampling, sim=sim)
    )

    return PrototypeAgent(
        planner,
        believe_manager
    )
