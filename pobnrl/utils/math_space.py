""" mathematical spaces or sets """

import numpy as np


class DiscreteSpace():
    """ DiscreteSpace discrete uninterupted space of some shape """

    def __init__(self, dim):
        """__init__ initiates a discrete space of size dim

        :param dim: is a list of dimensions
        """
        assert isinstance(dim, list)

        self.dim = np.array(dim)
        self.n = np.prod(self.dim)
        self.shape = self.dim.shape

    def dimensions(self) -> np.array:
        """dimensions dimensions returns the range of each dimension

        :rtype: np.array
        """
        return self.dim

    def sample(self) -> int:
        """sample returns a sample from the space at random

        :rtype: int
        """
        return (np.random.random(self.shape) * self.dim).astype(int)
