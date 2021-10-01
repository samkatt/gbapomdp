"""The tiger problem implemented as domain"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from typing import List, Optional

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)
from gym_gridverse.outer_env import OuterEnv

class GridVerseGymEnv(gym.Env):  
    def __init__(self, gamma: float,  config_file: str):

        inner_env = factory_env_from_yaml(config_file)

        obs_rep = DefaultObservationRepresentation(inner_env.observation_space)
        outer_env = OuterEnv(inner_env, observation_rep=obs_rep)

        self.core_env = GymEnvironment.from_environment(outer_env)

        self.action_space = self.core_env.action_space
        self.observation_space = self.core_env.observation_space
        self.discount = gamma

        self.seed()

    def seed(self, seed=None):
        self.core_env.seed(seed)

    def step(self, action: int):
        return self.core_env.step(action)

    def reset(self):
        return self.core_env.reset()

    def close(self):
        self.core_env.close()

    def render(self, mode='human'):
        self.core_env.render(mode)