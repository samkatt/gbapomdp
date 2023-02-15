import numpy as np
import random
import copy

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace

from general_bayes_adaptive_pomdps.domains.speedy_tool_delivery.macro_factored_all import ObjSearchDelivery_v4 as EnvToolDelivery

GET_TOOL_0 = 0
GET_TOOL_1 = 1
GET_TOOL_2 = 2
DELIVER_0 = 3
DELIVER_1 = 4

class ToolDeliveryV0(Domain):

    def __init__(self, human_speeds, render=False):
        """Construct domain
        Args:
        """
        super().__init__()

        self.core_env = EnvToolDelivery(human_speeds, render=render)
        self.human_speeds = human_speeds

        # STATE
        # x_coord, y_coord [2] [2]
        # current primitive timestep [max_step]
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs
        # which object is waited for - human 0: [n_objs]
        # which object is waited for - human 1: [n_objs]
        n_objs = 3
        max_timestep = 500
        self.n_objs = n_objs
        self._state_space = DiscreteSpace([2] + [2] + [max_timestep] + [3]
                                          + [3]*n_objs + [2]*n_objs
                                          + [n_objs] + [n_objs])

        self.coord_x_idx = 0
        self.coord_y_idx = 1
        self.timestep_idx = 2
        self.room_idx = 3

        # reduced STATE
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs
        # which object is waited for - human 0: [n_objs]
        # which object is waited for - human 1: [n_objs]
        self._rstate_space = DiscreteSpace([3]*n_objs +
                                           [2]*n_objs +
                                           [n_objs] +
                                           [n_objs])

        # OBSERVATION
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human 0 working step: [n_objs + 1] (only observable in the work-room)
        # human 1 working step: [n_objs + 1] (only observable in the work-room)
        self._obs_space = DiscreteSpace([3] + [3]*n_objs + [2]*n_objs +
                                        [n_objs + 1] + [n_objs + 1])

        # REDUCED-OBSERVATION
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human 0 working step: [n_objs + 1] (only observable in the work-room)
        # human 1 working step: [n_objs + 1] (only observable in the work-room)
        self._robs_space = DiscreteSpace([2]*n_objs + [n_objs + 1] + [n_objs + 1])

        # each agent has 5 possible actions: Get_Tool_i(0:n_objs - 1), Deliver_0, Deliver_1
        self._action_space = ActionSpace(n_objs + 2)

    @property
    def state_space(self) -> DiscreteSpace:
        return self._state_space

    @property
    def rstate_space(self) -> DiscreteSpace:
        return self._rstate_space

    @property
    def robservation_space(self) -> DiscreteSpace:
        return self._robs_space

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        return self._obs_space

    def get_state(self) -> np.ndarray:
        return self.core_env.get_state()

    # remove the room location and the tools that are carried
    def process_o_fcn(self, o):
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs (only observable in the tool-room)
        # human 0 working step: [n_objs + 1] (only observable in the work-room)
        # human 1 working step: [n_objs + 1] (only observable in the work-room)
        return o[:, 1 + self.n_objs:]

    def known_dyn_fcn(self, s, a, return_dist=False):
        return self.core_env.known_dyn_coord_fcn(s, a, return_dist)

    # remove coordinates, primitive timestep, room from the next state
    # STATE
    # x_coord, y_coord [2] [2]
    # current primitive timestep [max_step]
    # discrete room locations: [3]
    # which object in the basket: [3]*n_objs
    # which object are on the table: [2]*n_objs
    # which object is waited for - human 0: [n_objs]
    # which object is waited for - human 1: [n_objs]
    def process_ns_fcn(self, ns):
        return ns[:, 4:].long()

    # zeroing the current primitive timestep, coordinates
    def process_s_fcn(self, s):
        s_copy = copy.deepcopy(s)
        if len(s.shape) == 2:
            s_copy[:, self.timestep_idx] = 0
            s_copy[:, self.coord_x_idx] = 0
            s_copy[:, self.coord_y_idx] = 0

            return s_copy.long()

        elif len(s.shape) == 1:
            s_copy[self.timestep_idx] = 0
            s_copy[self.coord_x_idx] = 0
            s_copy[self.coord_y_idx] = 0

            return s_copy.astype(int)

        else:
            raise NotImplementedError

    def sample_start_state(self) -> np.ndarray:
        """returns the (deterministic) start state

        Args:

        RETURNS (`np.ndarray`):

        """

        # STATE
        # x_coord, y_coord [2] [2]
        # current primitive timestep [max_step]
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs
        # which object is waited for - human 0: [n_objs]
        # which object is waited for - human 1: [n_objs]

        init_state = np.zeros(self.state_space.ndim)
        return np.array(init_state, dtype=int)

    def reset(self) -> np.ndarray:
        """reset the environment

        Args:

        RETURNS (`np.ndarray`)

        """
        return self.core_env.reset()

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """a known reward function

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`): the reward of the transition

        """
        # STATE
        # x_coord, y_coord [2] [2]
        # current primitive timestep [max_step]
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs
        # which object is waited for - human 0: [n_objs]
        # which object is waited for - human 1: [n_objs]
        assert len(state) == (2*self.n_objs + 6), f"Len state {len(state)} is wrong"

        delta_time = new_state[self.timestep_idx] - state[self.timestep_idx]
        reward = -delta_time

        # faster human waiting incurs more penalty
        if self.human_speeds[0] < self.human_speeds[1]:
            human0_wait_penalty = -30
            human1_wait_penalty = -10
        else:
            human0_wait_penalty = -10
            human1_wait_penalty = -30

        objs_in_basket = state[4:4+self.n_objs]
        next_objs_in_basket = new_state[4:4+self.n_objs]

        # human 0
        curr_human0_needed_obj = int(state[-2])

        deliver_success = (action == self.n_objs) and \
                          (sum(next_objs_in_basket) < sum(objs_in_basket))

        # Deliver a good tool for human 0
        if objs_in_basket[curr_human0_needed_obj] > 0 and deliver_success:
            reward += 100
        else:
            reward += human0_wait_penalty

        # human 1
        curr_human1_needed_obj = int(state[-1])

        deliver_success = (action == self.n_objs + 1) and \
                          (sum(next_objs_in_basket) < sum(objs_in_basket))

        # Deliver a good tool for human 1
        if objs_in_basket[curr_human1_needed_obj] > 0 and deliver_success:
            reward += 100
        else:
            reward += human1_wait_penalty

        return reward

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if reached end in `new_state`

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`): whether the transition is terminal

        """
        # STATE
        # x_coord, y_coord [2] [2]
        # current primitive timestep [max_step]
        # discrete room locations: [3]
        # which object in the basket: [3]*n_objs
        # which object are on the table: [2]*n_objs
        # which object is waited for - human 0: [n_objs]
        # which object is waited for - human 1: [n_objs]

        done = False

        objs_in_basket = new_state[4:4+self.n_objs]
        objs_in_table = new_state[4+self.n_objs:4+2*self.n_objs]

        # the basket is empty and all objects are not on the table
        if sum(objs_in_basket) == 0 and sum(objs_in_table) == self.n_objs:
            done = True

        return done

    def step(self, action: int) -> DomainStepResult:
        s = self.core_env.get_state()
        next_obs, _, _, _ = self.core_env.step([action])
        next_s = self.core_env.get_state()
        r = self.reward(s, action, next_s)
        t = self.terminal(s, action, next_s)
        return DomainStepResult(next_obs[0], r, t)

class ToolDeliveryV0Prior(DomainPrior):
    """perfect prior for now
    """

    def __init__(self):
        """initiate the prior

        Args:
        """
        super().__init__()

        self.speed_list = [10, 20, 30]

    def sample(self, prior=None) -> Domain:
        """
        Prior that the correct order will start with the first tool
        """
        if prior:
            random_speeds = prior
        else:
            random_speeds = random.choices(self.speed_list, k=2)  # random w/ replacements
        return ToolDeliveryV0(human_speeds=random_speeds)

if __name__ == "__main__":
    env = ToolDeliveryV0(human_speeds=[10, 20], render=False)
    env.reset()

    index_2_str = ['Get_Tool_0', 'Get_Tool_1', 'Get_Tool_2', 'Deliver_0', 'Deliver_1']

    def action_to_str(action_idx):
        return f"Action: {index_2_str[action_idx]}"

    optimal = True

    rewards = []

    delivers = [DELIVER_0, DELIVER_1]

    if optimal:

        # act_list = [GET_TOOL_0] + [GET_TOOL_1] + [DELIVER_1] + [DELIVER_1] + [DELIVER_1] + [DELIVER_1]
                #  + [GET_TOOL_1] + [GET_TOOL_1] + [DELIVER_1] + [DELIVER_0]\
                #  + [GET_TOOL_2] + [GET_TOOL_2] + [DELIVER_1] + [DELIVER_0]

        act_list = [GET_TOOL_0, GET_TOOL_0] + delivers + \
                   [GET_TOOL_1, GET_TOOL_1] + delivers + \
                   [GET_TOOL_2, GET_TOOL_2] + delivers

        for action in act_list:
            print(action_to_str(action))
            state = env.step(action)
            print("Timestep:", env.get_state())
            print(state.observation, state.reward, state.terminal)
            print()
            rewards.append(state.reward)

    print(rewards)
    discounted_return = sum(
        pow(0.95, i) * r for i, r in enumerate(rewards)
    )

    print(optimal, discounted_return)
