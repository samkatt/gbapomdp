""" DQRN implementation """

from agents.agent import Agent

class DRQN(Agent):
    """ DRQN implementation """

    def __init__(self):
        self.last_observation = None
        self.last_action = None

    def reset(self):
        """ resets temporal information, not network """
        self.last_observation = None
        self.last_action = None

    def select_action(self):
        """ selects an action according to its network """

        # store we performed this action
        raise NotImplementedError()

    def update(self, _observation, _reward, _terminal):
        """ stores interaction and learns network """
        assert self.last_action is not None

        raise NotImplementedError()
