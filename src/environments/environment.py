""" environment interface """

import abc

class Environment(abc.ABC):
    """ interface to all environments """

    @abc.abstractmethod
    def reset(self):
        """ resets internal state to start a new episode """
        pass

    @abc.abstractmethod
    def step(self, _action):
        """ update state wrt action """
        pass
