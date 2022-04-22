from general_bayes_adaptive_pomdps.domains.small_box_pushing import (
    SmallBoxPushing,
)
import time

env = SmallBoxPushing()

print(env.reset())
print(env.action_space)

for i in range(100):
    _, _, done, _ = env.step([env.action_space_sample(0), env.action_space_sample(1)])
    env.render()
    time.sleep(1)

    print(env.get_state())

    if done:
        env.reset()

