""" environment interface """

import abc

from misc import DiscreteSpace


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

    @property
    @abc.abstractmethod
    def state(self):
        """ returns the current state """

    @property
    @abc.abstractmethod
    def action_space(self) -> DiscreteSpace:
        """ returns size of domain action space

        RETURNS (`pobnrl.misc.DiscreteSpace`): the action space

        """

    @property
    @abc.abstractmethod
    def observation_space(self) -> DiscreteSpace:
        """ returns size of domain observation space

        RETURNS (`pobnrl.misc.DiscreteSpace`): the observation space

        """

    def __repr__(self):
        return (f"{self.__class__} with action space {self.action_space}, "
                f"observation space {self.observation_space} and "
                f"current state {self.state}")
