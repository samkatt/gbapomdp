""" agents in POBNRL """

from enum import Enum

from environments import Simulator

from .agent import Agent, RandomAgent
from .model_free_agents import create_agent as create_mf_agent
from .model_based_agents import create_learning_agent, create_planning_agent


class AgentType(Enum):
    """ AgentType """
    MODELFREE = 1
    MODELBASED = 2
    PLANNING = 3


def create_agent(sim: Simulator, conf, agent_type: AgentType) -> Agent:
    """ factory function to construct agents

    Args:
         sim: (`pobnrl.environments.Simulator`):
         conf: (`namespace`): configurations

    RETURNS (`pobnrl.agents.agent.Agent`)

    """

    if conf.random_policy:
        return RandomAgent(sim.action_space)

    if agent_type == AgentType.MODELFREE:
        return create_mf_agent(sim.action_space, sim.observation_space, conf)

    if agent_type == AgentType.MODELBASED:
        return create_learning_agent(sim, conf)

    if agent_type == AgentType.PLANNING:
        return create_planning_agent(sim, conf)

    raise ValueError("Unknown agent type provided")
