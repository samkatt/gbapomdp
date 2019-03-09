""" environment interface """

import abc


class Environment(abc.ABC):
    """ interface to all environments """

    @abc.abstractmethod
    def reset(self):
        """ resets internal state and return first observation """

    @abc.abstractmethod
    def step(self, action) -> list:
        """ update state as a result of action

        Args:
             action: agent's taken action

        RETURNS (`list`): [observation, reward (float), terminal (bool)]

        """

    @abc.abstractmethod
    def spaces(self) -> dict:
        """ returns size of domain space {'O', 'A'}

        RETURNS (`dict`): {'O', 'A'} of spaces to sample from

        """
