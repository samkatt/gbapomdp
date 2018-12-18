""" cartpole environment """

from environments.environment import Environment

import gym

class Cartpole(Environment):
    """ cartpole environment """

    def __init__(self):
        self.cartpole = gym.make('CartPole-v0')
        self.cartpole = gym.wrappers.Monitor(self.cartpole, 'videos/', force=True)

    def __del__(self):
        self.cartpole.close()

    def reset(self):
        """ resets the cartpole gym environment """
        return self.cartpole.reset()

    def step(self, action):
        """ performs a step in the cartpole env """
        obs, reward, terminal, _ = self.cartpole.step(action)
        return obs, reward, terminal

    def spaces(self):
        """ returns spaces from cartpole gym env """
        return {"A": self.cartpole.action_space, "O": self.cartpole.observation_space}
