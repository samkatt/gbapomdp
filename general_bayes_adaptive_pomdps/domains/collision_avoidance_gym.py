"""The tiger problem implemented as domain"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from typing import List, Optional

from general_bayes_adaptive_pomdps.domains.collision_avoidance import (
    CollisionAvoidance,
)

class CollisionAvoidanceGymEnv(gym.Env):  
    def __init__(self, gamma: float,  domain_size: int):

        self.core_env = CollisionAvoidance(domain_size)

        # TODO:
        self.action_space = spaces.Discrete(self.core_env.action_space.n)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ))

        self.discount = gamma

        self.seed()

    def seed(self, seed=None):
        self.core_env.seed(seed)

    def step(self, action: int):
        step_result = self.core_env.step(action)

        return step_result.observation, step_result.reward, step_result.terminal, {}

    def reset(self):
        return self.core_env.reset()

    def close(self):
        pass