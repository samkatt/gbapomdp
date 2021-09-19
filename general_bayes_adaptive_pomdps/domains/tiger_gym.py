"""The tiger problem implemented as domain"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from typing import List, Optional

from general_bayes_adaptive_pomdps.domains.tiger import (
    Tiger,
)

class TigerGymEnv(gym.Env):  
    def __init__(self, gamma: float, one_hot_encode_observation: bool, correct_obs_probs: Optional[List[float]] = None):

        self.core_env = Tiger(one_hot_encode_observation, correct_obs_probs=correct_obs_probs)

        # TODO:
        self.action_space = spaces.Discrete(self.core_env.action_space.n + 1)
        self.observation_space = spaces.Box(0, 1, shape=(2, ))

        self.eps_rewards = []

        self.discount = gamma

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        if action in [2, 3]:
            action = 2

        step_result = self.core_env.step(action)

        self.eps_rewards.append(step_result.reward)

        # ep_info = {"is_success": 0}
        # if step_result.terminal:
        #     discounted_return = sum(pow(self.gamma, i) * r for i, r in enumerate(self.eps_rewards))
        #     ep_info["ep_return"] = discounted_return

        # print(action)

        return step_result.observation, step_result.reward, step_result.terminal, {}

    def reset(self):
        self.eps_rewards = []

        return self.core_env.reset()

    def close(self):
        pass