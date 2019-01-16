""" the interface all Q-networks should adhere to """

import abc

class QNetInterface(abc.ABC):
    """ interface to all environments """

    @abc.abstractmethod
    def reset(self):
        """ resets internal state and return first observation """
        pass

    @abc.abstractmethod
    def is_recurrent(self):
        """ update state wrt action. return: obs, reward, terminal """
        pass

    @abc.abstractmethod
    def qvalues(self, observation):
        """ returns the q values associated with the observations """
        pass

    @abc.abstractmethod
    def batch_update(self, obs, actions, rewards, next_obs, done_mask):
        """ performs a batch update """
        pass

    # [fixme] maybe not assume that there is a target net?
    @abc.abstractmethod
    def update_target(self):
        """ updates the target network """
        pass
