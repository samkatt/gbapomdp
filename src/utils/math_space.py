""" mathematical spaces or sets """

import numpy as np

class DiscreteSpace():
    """ discrete uninterupted space of some shape """

    def __init__(self, dim):
        assert isinstance(dim,list)

        self.dim = np.array(dim)
        self.n = np.prod(self.dim)
        self.shape = self.dim.shape

    def dimensions(self):
        """ returns the range of each dimension """
        return self.dim

    def sample(self):
        """ returns element at random """
        return (np.random.random(self.shape) * self.dim).astype(int)
