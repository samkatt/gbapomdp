from general_bayes_adaptive_pomdps.domains import *
import gym
import time

env = gym.make('grid-verse-v0')

env.reset()

for i in range(500):
    action = env.action_space.sample()
    env.step(action)
    env.render()
    time.sleep(1)