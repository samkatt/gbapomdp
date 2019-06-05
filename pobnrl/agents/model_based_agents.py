""" agents that act by learning a model of theenvironments """

from functools import partial
from typing import Any, Callable
import numpy as np

from environments import Simulator, SimulationResult
from domains.learned_environments import NeuralEnsemble
from misc import POBNRLogger

from .agent import Agent
from .planning.particle_filters import BeliefManager, rejection_sampling
from .planning.particle_filters import ParticleFilter, FlatFilter
from .planning.pouct import POUCT


class PrototypeAgent(Agent, POBNRLogger):
    """ default model-based agent """

    def __init__(
            self,
            planner: POUCT,
            belief_manager: BeliefManager
    ):
        """ creates this agent with planner and belief manager

        Args:
             planner: (`pobnrl.agents.planning.pouct.POUCT`):
             BeliefManager: (`pobnrl.agents.planning.particle_filters.BeliefManager`):

        """

        POBNRLogger.__init__(self)

        self._planner = planner
        self._belief_manager = belief_manager

        self._last_action = -1

    def reset(self):
        """ resets belief """

        self._belief_manager.reset()

    def episode_reset(self, _observation: np.ndarray):
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


class RejectionSamplingBelieveManager(BeliefManager):
    """ believe manager that uses rejection sampling """

    def __init__(
            self,
            num_particles: int,
            sim: Simulator,
            reset_particle_f: Callable[[Any], Any] = None):
        """ creates a believe manager based on rejection sampling

        Args:
             num_particles: (`int`): the number of particles in the filter
             sim: (`pobnrl.environments.Simulator`): the simulator to update particles
             reset_particle_f: (`Callable[[Any], Any]`): how to reset particles

        """

        super().__init__(
            num_particles=num_particles,
            filter_type=FlatFilter,
            sample_particle_f=sim.sample_start_state,
            update_belief_f=partial(self.update_belief, env=sim),
            reset_particle_f=reset_particle_f
        )

    @staticmethod
    def update_belief(
            particle_filter: ParticleFilter,
            action: int,
            observation: np.ndarray,
            env: Simulator) -> ParticleFilter:
        """ Applies belief rejection sampling

        Will update the belief by simulating a step in the simulator and using
        rejection sampling on the observation

        Args:
        particle_filter: (`pobnrl.agents.planning.particle_filters.ParticleFilter`): current belief
        env: (`pobnrl.environments.Simulator`): simulator as a dynamic model
        action: (`int`): taken action
        observation: (`np.ndarray`): perceived observation

        RETURNS (`pobnrl.agents.planning.particle_filters.ParticleFilter`): new belief

        """

        update_step = partial(env.simulation_step, action=action)

        def extract_state(interaction: SimulationResult) -> Any:
            return interaction.state

        def observation_equals(interaction: SimulationResult) -> bool:
            return np.all(interaction.observation == observation)

        return rejection_sampling(
            particle_filter,
            process_sample_f=update_step,
            accept_f=observation_equals,
            extract_particle_f=extract_state,
        )


def create_learning_agent(env: Simulator, conf) -> PrototypeAgent:
    """ factory function to construct model based learning agents

    Args:
         env: (`pobnrl.environments.Simulator`) simulator
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.model_based_agents.PrototypeAgent`)

    """

    if conf.belief != 'rejection_sampling':
        raise ValueError('belief must be rejection_sampling')

    planner = POUCT(
        env,
        conf.num_sims,
        conf.exploration,
        conf.horizon,
        conf.gamma
    )

    def reset_particle(augmented_state):
        NeuralEnsemble.AugmentedState(
            env.sample_start_state(),
            augmented_state.model
        )

    believe_manager = RejectionSamplingBelieveManager(
        num_particles=conf.num_particles,
        sim=env,
        reset_particle_f=reset_particle
    )

    return PrototypeAgent(
        planner,
        believe_manager
    )


def create_planning_agent(env: Simulator, conf) -> PrototypeAgent:
    """ factory function to construct planning agents

    Args:
         env: (`pobnrl.environments.Simulator`) simulator
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.model_based_agents.PrototypeAgent`)

    """

    if conf.belief != 'rejection_sampling':
        raise ValueError('belief must be rejection_sampling')

    planner = POUCT(
        env,
        conf.num_sims,
        conf.exploration,
        conf.horizon,
        conf.gamma
    )

    believe_manager = RejectionSamplingBelieveManager(
        conf.num_particles,
        env
    )

    return PrototypeAgent(
        planner,
        believe_manager
    )
