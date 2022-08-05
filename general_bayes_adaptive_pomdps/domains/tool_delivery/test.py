from single_human import ObjSearchDelivery_v4 as dev_env
from original import ObjSearchDelivery_v4 as or_env
import time

env = dev_env()

env.reset()

env.render()

for i in range(5):
    env.step([1, 0])
    env.render()
    time.sleep(1)

for i in range(9):
    env.step([0, 0])
    env.render()
    time.sleep(1)

for i in range(8):
    env.step([4, 0])
    env.render()
    time.sleep(1)
# env.step([0, 0])
# env.render()


# env.step([0, 0])
# env.render()



time.sleep(200)
