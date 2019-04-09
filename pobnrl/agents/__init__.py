""" agents in POBNRL """

from misc import PiecewiseSchedule, FixedExploration, DiscreteSpace

from .networks import create_qnet
from .agent import Agent, RandomAgent
from .model_free_agents import BaselineAgent, EnsembleAgent


def create_agent(
        action_space: DiscreteSpace,
        observation_space: DiscreteSpace,
        conf) -> Agent:
    """ factory function to construct agents

    Args:
         action_space: (`pobnrl.misc.DiscreteSpace`): of environment
         observation_space: (`pobnrl.misc.DiscreteSpace`) of environment
         conf: (`namespace`) configurations

    RETURNS (`pobnrl.agents.agent.Agent`)

    """

    if conf.random_policy:
        return RandomAgent(action_space)

    if 1 >= conf.exploration >= 0:
        exploration_schedule = FixedExploration(conf.exploration)
    else:
        exploration_schedule = PiecewiseSchedule(
            [(0, 1.0), (2e4, 0.1), (1e5, 0.05)], outside_value=0.05
        )

    if conf.num_nets == 1:
        # single-net agent

        return BaselineAgent(
            create_qnet(
                action_space,
                observation_space,
                'q_net',
                conf
            ),
            action_space,
            exploration_schedule,
            conf
        )

    # num_nets > 1: ensemble agent

    def qnet_constructor(name: str):
        return create_qnet(
            action_space,
            observation_space,
            name,
            conf
        )

    return EnsembleAgent(
        qnet_constructor,
        action_space,
        exploration_schedule,
        conf,
    )
