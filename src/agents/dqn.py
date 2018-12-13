""" dqn implementation """

from agents.agent import Agent

class DQN(Agent):
    """ the agent that uses DQN """

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

class DQNBrett(Agent):
    """ DQN implementation almost directly taken from Brett """

    def __init__(self):
        """ TODO """
        raise NotImplementedError()

    def reset(self):
        """ TODO """
        raise NotImplementedError()

    def select_action(self):
        """ TODO """
        raise NotImplementedError()

    def update(self, _observation, _reward, _terminal):
        """ TODO """
        raise NotImplementedError()


class DQNLuke(Agent):
    """ DQN implementation almost directly taken from Brett """

    def __init__(self):
        """ TODO """
        raise NotImplementedError()

    def reset(self):
        """ TODO """
        raise NotImplementedError()

    def select_action(self):
        """ TODO """
        raise NotImplementedError()

    def update(self, _observation, _reward, _terminal):
        """ TODO """
        raise NotImplementedError()
