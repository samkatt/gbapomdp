""" cartpole environment """

import time
import gym

from environments.environment import Environment


class Cartpole(Environment):
    """ cartpole environment """

    def __init__(self, verbose: bool):
        """ constructs cartpole with optionally graphical representation

        Args:
             verbose: (`bool`): whether to be (graphically) verbose

        """

        self._cur_time = 0

        recording_policy = self.show_recording if verbose else False

        self.cartpole = gym.make('CartPole-v0')
        self.cartpole = gym.wrappers.Monitor(
            self.cartpole,
            'videos/',
            force=True,
            video_callable=recording_policy)

    def show_recording(self, _):
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

    def step(self, action) -> list:
        """ update state as a result of action

        Uses gym.cartpole for the actual transition

        Args:
             action: agent's taken action

        RETURNS (`list`): [observation, reward (float), terminal (bool)]

        """

        obs, reward, terminal, _ = self.cartpole.step(action)
        return obs, reward, terminal

    def spaces(self) -> dict:
        """ returns size of domain space {'O', 'A'}

        returns aciton and observation space of the gym.carpole environment

        RETURNS (`dict`): {'O', 'A'} of spaces to sample from

        """
        return {"A": self.cartpole.action_space,
                "O": self.cartpole.observation_space}
