""" cartpole environment """

import time

from environments.environment import Environment

import gym

class Cartpole(Environment):
    """ cartpole environment """

    def __init__(self, conf):

        self._cur_time = 0

        recording_policy = self._show_recording if conf.verbose else False

        self.cartpole = gym.make('CartPole-v0')
        self.cartpole = gym.wrappers.Monitor(
            self.cartpole,
            'videos/',
            force=True,
            video_callable=recording_policy)

    def _show_recording(self, _):
        """ returns whether a recording should be shown """
        if time.time() - self._cur_time > 20:
            self._cur_time = time.time()
            print('showing recording..')
            return True

        return False


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
