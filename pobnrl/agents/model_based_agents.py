""" agents that act by learning a model of the environment """

from typing import Any
import numpy as np

from agents.agent import Agent
from environments.environment import Environment, EnvironmentInteraction

from .planning.beliefs import BeliefManager, rejection_sampling
from .planning.beliefs import WeightedFilter, FlatFilter
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
             BeliefManager: (`pobnrl.agents.planning.beliefs.BeliefManager`):

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

        self._last_action \
            = self._planner.select_action(self._belief_manager.belief)

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
        env: Environment,
        action: int,
        observation: np.ndarray):
    """ TODO """

    def env_step(state: Any):
        """ TODO """
        print(f"stepping from state {state} with action {action}")
        env.state = state
        return env.step(action)

    def observation_equals(interaction: EnvironmentInteraction):
        """ TODO """
        print(f"comparing {observation} with {interaction}")
        return np.all(interaction.observation == observation)

    return rejection_sampling(
        particle_filter,
        env_step,
        observation_equals,
        lambda interaction: interaction.state
    )


def belief_importance_sampling(
        particle_filter: FlatFilter,
        env: Environment,
        action: int,
        observation: np.ndarray):
    """ TODO """

    raise NotImplementedError(
        "missing P(o|s,a) from environments to do importance sampling"
    )


def create_agent(env: Environment, conf) -> PrototypeAgent:
    """ factory function to construct planning agents

    Args:
         env: (`pobnrl.environments.environment.Environment`) of environment
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.model_based_agents.PrototypeAgent`)

    """

    if conf.belief == 'rejection_sampling':

        belief_type = FlatFilter

        def update_belief_f(
                belief: FlatFilter,
                action: int,
                observation: np.ndarray):
            return belief_rejection_sampling(belief, env, action, observation)

    elif conf.belief == 'importance_sampling':

        belief_type = WeightedFilter

        def update_belief_f(
                belief: WeightedFilter,
                action: int,
                observation: np.ndarray):
            return belief_importance_sampling(belief, env, action, observation)

    else:
        raise ValueError(
            'belief must be either rejection or importance sampling'
        )

    belief_manager = BeliefManager(
        env.sample_start_state,
        belief_type,
        update_belief_f,
        conf
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
