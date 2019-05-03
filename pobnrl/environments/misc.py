""" miscellaneous functions for environments """

import numpy as np

from pobnrl.misc import DiscreteSpace


class ActionSpace(DiscreteSpace):
    """ action space for environments

    TODO: add one_hot function

    """

    def __init__(self, size: int):
        """ initiates an action space of size

        Args:
             dim: (`int`): number of actions

        """
        super().__init__([size])

    def sample(self) -> np.array:
        """ returns a sample from the space at random

        RETURNS (`np.array`): a sample in the space of this

        """
        return super().sample()[0]

    def __repr__(self):
        return f"ActionSpace of size {self.n}"
