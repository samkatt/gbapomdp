""" cartpoleenvironments """

import time

import gym

from environments import Environment, EnvironmentInteraction, ActionSpace
from misc import DiscreteSpace, POBNRLogger, LogLevel


class Cartpole(Environment):
    """ cartpole environment """

    logger = POBNRLogger(__name__)

    def __init__(self, verbose: bool):
        """ constructs cartpole with optionally graphical representation

        Args:
             verbose: (`bool`): whether to be (graphically) verbose

        """

        self._cur_time = 0

        recording_policy = self.show_recording if verbose else False

        self.cartpole = gym.make('CartPole-v0')

        # because of mysteries of python self.cartpole must be initiated first
        self.cartpole = gym.wrappers.Monitor(
            self.cartpole,
            'videos/',
            force=True,
            video_callable=recording_policy
        )

    def show_recording(self, _):
        """ returns whether a recording should be shown """
        if time.time() - self._cur_time > 20:
            self._cur_time = time.time()
            self.logger.log(LogLevel.V2, 'showing recording..')
            return True

        return False

    def __del__(self):
        self.cartpole.env.close()

    def reset(self):
        """ resets the cartpole gym environment """
        return self.cartpole.reset()

    def step(self, action: int) -> EnvironmentInteraction:
        """ update state as a result of action

        Uses gym.cartpole for the actual transition

        Args:
             action: (`int`): agent's taken action

        RETURNS (`pobnrl.environments.EnvironmentInteraction`): the transition

        """

        obs, reward, terminal, _ = self.cartpole.step(action)
        return EnvironmentInteraction(obs, reward, terminal)

    @property
    def action_space(self) -> ActionSpace:
        """ the underlying open ai gyme cartpole action space """
        return self.cartpole.action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """ the underlying open ai gyme cartpole observation space """
        return self.cartpole.observation_space
