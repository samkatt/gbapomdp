""" miscellaneous functions for environments """

import numpy as np

from misc import DiscreteSpace


class ActionSpace():
    """ action space for environments """

    def __init__(self, size: int):
        """ initiates an action space of size

        Args:
             dim: (`List[int]`): is a list of dimensions

        """
        assert size > 1

        self._space = DiscreteSpace([size])

    @property
    def n(self) -> int:  # pylint: disable=invalid-name
        """ Number of elements in space

        While the naming is pretty awful, it is consistent with the `Space`
        class of open AI gym, which I prioritized here

        RETURNS (`int`):

        """
        return self._space.n

    @property
    def dimensions(self) -> np.array:
        """ returns the range of each dimension

        RETURNS (`np.array`): each member is the size of its dimension

        """
        return self._space.dimensions

    @property
    def shape(self) -> tuple:
        """ returns the shape of the space

        Args:

        RETURNS (`tuple`): as like np.shape

        """
        return self._space.shape

    def sample(self) -> np.array:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return self._space.sample()[0]

    def __repr__(self):
        return f"ActionSpace of size {self.n}"
