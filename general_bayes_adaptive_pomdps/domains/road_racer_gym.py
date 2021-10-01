"""The tiger problem implemented as domain"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from typing import List, Optional

from general_bayes_adaptive_pomdps.domains.road_racer import (
    RoadRacer,
)

class RoadRacerGymEnv(gym.Env):  
    def __init__(self, gamma: float, domain_size: int,):

        lane_probs = np.arange(1, domain_size + 1) / (domain_size + 1)

        self.core_env = RoadRacer(lane_probs)

        # TODO:
        self.action_space = spaces.Discrete(self.core_env.action_space.n)
        self.observation_space = spaces.Box(0, 1, shape=(6, ))

        self.discount = gamma

        self.seed()

    def seed(self, seed=None):
        self.core_env.seed(seed)

    def step(self, action: int):
        step_result = self.core_env.step(action)

        return self._to_one_hot(step_result.observation), step_result.reward, step_result.terminal, {}

    def reset(self):
        return self._to_one_hot(self.core_env.reset())

    def close(self):
        pass

    def _to_one_hot(self, obs):
        return np.eye(6)[obs[0]]
