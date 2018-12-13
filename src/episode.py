""" implements running an episode """

def run_episode(env, agent, conf):
    """ runs a single episode """

    terminal = False
    discounted_return = 0
    discount = conf.discount

    env.reset()
    agent.reset()
    while not terminal:

        action = agent.select_action()

        observation, reward, terminal = env.step(action)

        agent.update(observation, reward, terminal)

        discounted_return += discount * reward
        discount *= conf.discount

    return discounted_return
