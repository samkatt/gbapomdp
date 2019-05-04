""" agents in POBNRL """

from pobnrl.environments.environment import Environment

from .agent import Agent, RandomAgent
from .model_free_agents import create_agent as create_mf_agent
from .model_based_agents import create_agent as create_mb_agent


def create_agent(env: Environment, conf) -> Agent:
    """ factory function to construct agents

    Args:
         env: (`pobnrl.environments.environment.Environment`) of environment
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.agent.Agent`)

    """

    if conf.random_policy:
        return RandomAgent(env.action_space)

    if conf.agent_type == "model-free":
        return create_mf_agent(env.action_space, env.observation_space, conf)

    if conf.agent_type == "planning":
        return create_mb_agent(env.action_space, env, conf)

    raise ValueError("Unknown agent type provided")
