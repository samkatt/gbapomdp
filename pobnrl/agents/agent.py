""" agent interface """
import abc


class Agent(abc.ABC):
    """ all agents must implement this interface """

    @abc.abstractmethod
    def reset(self, obs):
        """ called after each episode to prepare for the next

        Args:
             obs: the observation of the start of the episode

        """

    @abc.abstractmethod
    def select_action(self):
        """ asks the agent to select an action """

    @abc.abstractmethod
    def update(self, _observation, _reward: float, _terminal: bool):
        """ calls at the end of a real step to allow the agent to update

        The provided observation, reward and terminal are the result of a step
        in the real world given the last action

        Args:
             _observation: the previous observation of the step
             _reward: (`float`): the reward associated with the last step
             _terminal: (`bool`): whether the last step was terminal

        """
