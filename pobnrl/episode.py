""" implements running an episode """


def run_episode(env, agent, conf) -> float:
    """run_episode runs a single episode

    :param env: the environment to run in bla `pobnrl.environments.environment.Environment`
    :param agent: the agent that learns and takes decisions
    :param conf: configurations

    :rtype: float discounted return
    """

    terminal = False
    discounted_return = 0
    discount = conf.gamma

    obs = env.reset()
    agent.reset(obs)
    time = 0
    while not terminal and time < conf.horizon:

        action = agent.select_action()

        observation, reward, terminal = env.step(action)

        agent.update(observation, reward, terminal)

        discounted_return += discount * reward
        discount *= conf.gamma

        time += 1

    return discounted_return
