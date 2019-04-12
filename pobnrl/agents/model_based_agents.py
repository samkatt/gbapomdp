""" agents that act by learning a model of the environment """

import numpy as np

from agents import Agent


class PrototypeAgent(Agent):
    """ default model-based agent """

    def reset(self):
        """ resets agent to initial state

        TODO: add doc
        TODO: implement

        """
        raise NotImplementedError

    def episode_reset(self, observation: np.ndarray):
        """ called after each episode to prepare for the next

        TODO: add doc
        TODO: implement

        Args:
             observation: (`np.ndarray`): the initial episode observation

        """
        raise NotImplementedError

    def select_action(self) -> int:
        """ asks the agent to select an action

        TODO: add doc
        TODO: implement

        RETURNS: action

        """
        raise NotImplementedError

    def update(
            self,
            observation: np.ndarray,
            reward: float,
            terminal: bool):
        """ calls at the end of a real step to allow the agent to update

        The provided observation, reward and terminal are the result of a step
        in the real world given the last action

        TODO: add doc
        TODO: implement

        Args:
             observation (`np.ndarray`): the observation
             reward: (`float`): the reward associated with the last step
             terminal: (`bool`): whether the last step was terminal

        """
        raise NotImplementedError
