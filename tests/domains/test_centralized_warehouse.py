import numpy as np
import pytest
import time

from general_bayes_adaptive_pomdps.domains.warehouse.ma_single_room import (
    ObjSearchDelivery_v4,
)

env = ObjSearchDelivery_v4()

env.reset()

for i in range(100):
    env.step([2, 0, 1])
    time.sleep(1)
    env.render()