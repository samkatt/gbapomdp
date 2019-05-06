""" agents in POBNRL """

from .agent import Agent, RandomAgent
from .model_free_agents import create_agent as create_mf_agent
from .model_based_agents import create_agent as create_mb_agent


def create_agent(env, conf) -> Agent:
    """ factory function to construct agents

    TODO: refactor to allow typing:

    this function right now cannot be type checked because it assumes env can
    be used both as POUCTSimulator and as a Environment. Currently this is
    fine(ish), since this applies to most domain, but it can become and issue

    Args:
         env: () any domain that is both a Environment and a POUCTSimulator
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.agent.Agent`)

    """

    if conf.random_policy:
        return RandomAgent(env.action_space)

    if conf.agent_type == "model-free":
        return create_mf_agent(env.action_space, env.observation_space, conf)

    if conf.agent_type == "planning":
        return create_mb_agent(env, conf)

    raise ValueError("Unknown agent type provided")
