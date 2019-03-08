""" environment interface """

import abc


class Environment(abc.ABC):
    """ interface to all environments """

    @abc.abstractmethod
    def reset(self):
        """ resets internal state and return first observation """

    @abc.abstractmethod
    def step(self, action) -> list:
        """step update state wrt action

        :param action: which action was taken by the agent
        :rtype: list [observation, reward (float), terminal (bool)]
        """

    @abc.abstractmethod
    def spaces(self) -> dict:
        """spaces returns size of domain space {'O', 'A'}

        :rtype: dict {'O', 'A'} of spaces to sample from
        """
