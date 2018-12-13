""" agent interface """
import abc

class Agent(abc.ABC):
    """ agent interface """

    @abc.abstractmethod
    def reset(self):
        """ requests the agent to reset its internal state for a new episode """
        pass

    @abc.abstractmethod
    def select_action(self):
        """ asks the agent to select an action """
        pass

    @abc.abstractmethod
    def update(self, _observation, _reward, _terminal):
        """ informs agent of observed transition """
        pass
