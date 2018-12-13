""" tiger environment """

from random import randint
from environments.environment import Environment

class Tiger(Environment):
    """ the tiger environment """

    # pylint: disable=R0201
    def sample_start_state(self):
        """ returns a random state (tiger left or right) """
        return randint(0, 1)

    def __init__(self):
        self.state = self.sample_start_state()

    def reset(self):
        """ resets state """
        self.state = self.sample_start_state()

    def step(self, _action):
        """ update state wrt action (listen or open) """

        raise NotImplementedError()
