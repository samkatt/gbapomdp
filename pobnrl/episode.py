""" implements running an episode """

import pobnrl.agents.agent
import pobnrl.environments.environment


def run_episode(
        env: pobnrl.environments.environment.Environment,
        agent: pobnrl.agents.agent.Agent,
        conf) -> float:
    """ runs a single episode of the agent in the environmnet

    Args:
         env: (`pobnrl.environments.environment.Environment`):
         agent: (`pobnrl.agents.agent.Agent`):
         conf: configurations

    RETURNS (`float`): discounted return

    """

    obs = env.reset()
    agent.episode_reset(obs)

    discounted_return = 0
    discount = 1  # discount accumulates by multiplying with conf.gamma
    time = 0
    terminal = False

    while not terminal and time < conf.horizon:

        action = agent.select_action()

        step = env.step(action)

        agent.update(step.observation, step.reward, step.terminal)

        discounted_return += discount * step.reward
        discount *= conf.gamma

        terminal = step.terminal
        time += 1

    return discounted_return
