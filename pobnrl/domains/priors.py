""" priors over the domains """

import abc

from numpy.random import dirichlet

from environments import Simulator

from .tiger import Tiger


class Prior(abc.ABC):
    """ the interface to priors """

    @abc.abstractmethod
    def sample(self) -> Simulator:
        """ sample a simulator

        RETURNS (`pobnrl.environments.Simulator`):
        """


class TigerPrior(Prior):
    """ standard prior over the tiger domain

    The transition model is known, however the probability of observing the
    tiger correctly is not. Here we assume a Dir(6,4) belief over this
    distribution.

    """

    def __init__(self, use_one_hot: bool):
        """ initiate the prior, `use_one_hot = True` will make observation one-hot encoded"""

        self._use_one_hot = use_one_hot

    def sample(self) -> Simulator:
        """ returns a Tiger instance with some correct observation prob

        This prior over the observation probability is a Dirichlet with alpha
        [6,4]

        RETURNS (`Simulator`):

        """
        sampled_observation_prob = dirichlet([6, 4])[0]

        return Tiger(use_one_hot=self._use_one_hot, correct_obs_prob=sampled_observation_prob)
