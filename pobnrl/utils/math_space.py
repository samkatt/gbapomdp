""" mathematical spaces or sets """

import numpy as np


class DiscreteSpace():
    """ DiscreteSpace discrete uninterupted space of some shape """

    def __init__(self, dim: list[int]):
        """ initiates a discrete space of size dim

        Args:
             dim: (`list[int]`): is a list of dimensions

        """
        assert isinstance(dim, list)

        self.dim = np.array(dim)
        self.n = np.prod(self.dim)
        self.shape = self.dim.shape

    def dimensions(self) -> np.array:
        """ returns the range of each dimension

        RETURNS (`np.array`): array where each member indicates the size of its dimension

        """
        return self.dim

    def sample(self) -> np.array:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return (np.random.random(self.shape) * self.dim).astype(int)
