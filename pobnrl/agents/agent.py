""" agent interface """
import abc


class Agent(abc.ABC):
    """ agent interface """

    @abc.abstractmethod
    def reset(self, obs):
        """ requests the agent to reset its internal state with first observation"""
        pass

    @abc.abstractmethod
    def select_action(self):
        """ asks the agent to select an action """
        pass

    @abc.abstractmethod
    def update(self, _observation, _reward, _terminal):
        """update informs agent of observed transition

        :param _observation: the observation from the last step
        :param _reward: the reward of the last step
        :param _terminal: whether the last step was terminal
        """
        pass
