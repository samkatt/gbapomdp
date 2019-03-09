""" the interface all Q-networks should adhere to """

import abc


class QNetInterface(abc.ABC):
    """ interface to all Q networks """

    @abc.abstractmethod
    def reset(self):
        """ resets the internal state to prepare for a new episode """

    @abc.abstractmethod
    def is_recurrent(self) -> bool:
        """ returns whether this is recurrent

        May be useful to know w.r.t. an internal state

        RETURNS (`bool`):

        """

    @abc.abstractmethod
    def qvalues(self, observation):
        """ returns the Q-values associated with the observation (net input)

        Args:
             observation: the input to the network

        """

    @abc.abstractmethod
    def batch_update(self, obs, actions, rewards, next_obs, done_mask):
        """ performs a batch update on the network on the provided input

        Basically the stochastic gradient descent step where, for any i,
        obs[i], actions[i], rewards[i], next_obs[i] and done_mask[i] are respectively the
        observation, actions, reward, next observation and terminality of some step i

        Args:
             obs: the observations or input to the network
             actions: the chosen action
             rewards: the rewards associated with each obs-action pair
             next_obs: the next observation after taking action and seeing obs
             done_mask: a boolean for each transition showing whether the transition was terminal

        """

    # FIXME: maybe not assume that there is a target net?
    @abc.abstractmethod
    def update_target(self):
        """ updates the target network """
