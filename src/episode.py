""" implements running an episode """

def run_episode(env, agent, conf):
    """ runs a single episode """

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
