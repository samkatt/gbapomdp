""" agents that act by learning a model of the environment """

from functools import partial
from typing import Any
import numpy as np

from agents.agent import Agent
from environments.environment import Simulator, SimulatedInteraction

from .planning.particle_filters import BeliefManager, rejection_sampling
from .planning.particle_filters import FlatFilter
from .planning.pouct import POUCT


class PrototypeAgent(Agent):
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
        self._planner = planner
        self._belief_manager = belief_manager

        self._last_action = None

    def reset(self):
        """ resets belief """

        self._belief_manager.reset()

    def episode_reset(self, _observation: np.ndarray):
        """ resets belief

        Ignores observation for now

        Args:
             observation: (`np.ndarray`): the initial episode observation

        """

        self._belief_manager.reset()

    def select_action(self) -> int:
        """  runs PO-UCT on current belief

        RETURNS (`int`): action

        """

        self._last_action = self._planner.select_action(self._belief_manager.belief)

        return self._last_action

    def update(
            self,
            observation: np.ndarray,
            _reward: float,
            _terminal: bool):
        """ calls at the end of a real step to allow the agent to update

        Will update the belief given the observation (and last action)

        Args:
             observation (`np.ndarray`): the observation
             reward: (`float`): ignored
             terminal: (`bool`): ignored

        """

        self._belief_manager.update(self._last_action, observation)


def belief_rejection_sampling(
        particle_filter: FlatFilter,
        action: int,
        observation: np.ndarray,
        env: Simulator) -> FlatFilter:
    """ Applies belief rejection sampling

    Will update the belief by simulating a step in the simulator and using
    rejection sampling on the observation

    Args:
         particle_filter: (`pobnrl.agents.planning.particle_filters.FlatFilter`): current belief
         env: (`pobnrl.environments.environment.Simulator`): simulator as a dynamic model
         action: (`int`): taken action
         observation: (`np.ndarray`): perceived observation

    RETURNS (`pobnrl.agents.planning.particle_filters.FlatFilter`): new belief

    """

    env_step = partial(env.simulation_step, action=action)

    def extract_state(interaction: SimulatedInteraction) -> Any:
        return interaction.state

    def observation_equals(interaction: SimulatedInteraction) -> bool:
        return np.all(interaction.observation == observation)

    return rejection_sampling(
        particle_filter,
        process_sample_f=env_step,
        accept_f=observation_equals,
        extract_particle_f=extract_state,
    )


def create_agent(env: Simulator, conf) -> PrototypeAgent:
    """ factory function to construct planning agents

    TODO: implement importance_sampling

    Args:
         env: (`pobnrl.environments.environment.Simulator`) simulator
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.model_based_agents.PrototypeAgent`)

    """

    if conf.belief != 'rejection_sampling':
        raise ValueError('belief must be rejection_sampling')

    update_belief_f = partial(belief_rejection_sampling, env=env)

    belief_manager = BeliefManager(
        env.sample_start_state,
        FlatFilter,
        update_belief_f,
        conf.num_particles
    )

    planner = POUCT(
        env,
        conf.num_sims,
        conf.exploration,
        conf.horizon,
        conf.gamma
    )

    return PrototypeAgent(
        planner,
        belief_manager
    )
