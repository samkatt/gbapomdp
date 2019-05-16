""" agents in POBNRL """

from environments import POUCTSimulator

from .agent import Agent, RandomAgent
from .model_free_agents import create_agent as create_mf_agent
from .model_based_agents import create_agent as create_mb_agent


def create_agent(sim: POUCTSimulator, conf) -> Agent:
    """ factory function to construct agents

    Args:
         sim: (`pobnrl.environments.POUCTSimulator`):
         conf: (`namespace`): configurations

    RETURNS (`pobnrl.agents.agent.Agent`)

    """

    if conf.random_policy:
        return RandomAgent(sim.action_space)

    if conf.agent_type == "model-free":
        return create_mf_agent(sim.action_space, sim.observation_space, conf)

    if conf.agent_type == "planning":
        return create_mb_agent(sim, conf)

    raise ValueError("Unknown agent type provided")
