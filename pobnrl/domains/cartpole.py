""" cartpoleenvironments """

import time

import gym
import numpy as np

from environments import Environment, EnvironmentInteraction, ActionSpace, GymSpace
from misc import Space, POBNRLogger


class Cartpole(Environment, POBNRLogger):
    """ cartpole environment """

    def __init__(self) -> None:
        """ constructs cartpole with optionally graphical representation """

        POBNRLogger.__init__(self)

        self._cur_time = 0

        recording_policy = self.show_recording if self.log_is_on(POBNRLogger.LogLevel.V2) else False

        self.cartpole = gym.make('CartPole-v0')

        # because of mysteries of python self.cartpole must be initiated first
        self.cartpole = gym.wrappers.Monitor(
            self.cartpole,
            'videos/',
            force=True,
            video_callable=recording_policy
        )

        self._action_space = ActionSpace(2)
        self._observation_space = GymSpace(self.cartpole.observation_space)

    def show_recording(self, _):
        """ returns whether a recording should be shown """
        if time.time() - self._cur_time > 20:
            self._cur_time = time.time()
            self.log(POBNRLogger.LogLevel.V2, 'showing recording..')
            return True

        return False

    def __del__(self):
        self.cartpole.env.close()

    def reset(self) -> np.ndarray:
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
        return self._action_space

    @property
    def observation_space(self) -> Space:
        """ the underlying open ai gyme cartpole observation space """
        # FIXME: wrap gym spaces
        return self._observation_space  # type: ignore
