""" environment interface """

import abc

class Environment(abc.ABC):
    """ interface to all environments """

    @abc.abstractmethod
    def reset(self):
        """ resets internal state and return first observation """
        pass

    @abc.abstractmethod
    def step(self, _action):
        """ update state wrt action. return: obs, reward, terminal """
        pass

    @abc.abstractmethod
    def spaces(self):
        """ returns size of domain space {'O', 'A'} """
        pass
