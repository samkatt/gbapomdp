""" agents in POBNRL """

from misc import PiecewiseSchedule, NoExploration, DiscreteSpace

from .networks import create_qnet
from .agent import Agent, RandomAgent, BaselineAgent, EnsembleAgent


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

    if conf.num_nets == 1:

        return BaselineAgent(
            create_qnet(
                action_space,
                observation_space,
                'q_net',
                conf
            ),
            action_space,
            PiecewiseSchedule(
                [(0, 1.0), (2e4, 0.1), (1e5, 0.05)], outside_value=0.05
            ),
            conf
        )

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
        NoExploration(),
        conf,
    )
