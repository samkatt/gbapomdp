import numpy as np
import random
import time
import copy

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace

from general_bayes_adaptive_pomdps.domains.ordered_tool_delivery.macro_factored_all import ObjSearchDelivery_v4 as EnvToolDelivery

from itertools import permutations

GET_TOOL_0 = 0
GET_TOOL_1 = 1
GET_TOOL_2 = 2
DELIVER = 3

class ToolDeliveryV0(Domain):

    def __init__(self, correct_tool_order, obs_corr_prob=1.0, render=False):
        """Construct domain
        Args:
        """
        super().__init__()

        self.correct_tool_order = correct_tool_order
        self.core_env = EnvToolDelivery(correct_tool_order, obs_corr_prob, render=render)

        # STATE
        # x_coord, y_coord
        # current primitive timestep
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # human working step: [n_objs + 1]
        n_objs = 3
        max_timestep = 500
        self.n_objs = n_objs
        self._state_space = DiscreteSpace([2] + [2] + [max_timestep] + [2] + [2]*n_objs +
                                          [n_objs + 1])

        self.coord_x_idx = 0
        self.coord_y_idx = 1
        self.timestep_idx = 2
        self.room_idx = 3

        # reduced STATE
        # which object in the basket: [2]*n_objs
        # human working step: [n_objs + 1]
        self._rstate_space = DiscreteSpace([2]*n_objs +
                                           [n_objs + 1])

        # OBSERVATION
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # human working step: [n_objs + 1] (only observable in the work-room)
        self._obs_space = DiscreteSpace([2] + [2]*n_objs + [n_objs + 1])

        # REDUCED-OBSERVATION
        # human working step: [n_objs + 1] (only observable in the work-room)
        self._robs_space = DiscreteSpace([n_objs + 1])

        # each agent has 4 possible actions: Get_Tool_i(0:n_objs - 1), Deliver
        self._action_space = ActionSpace(n_objs + 1)

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
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # human working step: [n_objs + 1] (only observable in the work-room)
        return o[:, 1 + self.n_objs:]

    def known_dyn_fcn(self, s, a, return_dist=False):
        return self.core_env.known_dyn_coord_fcn(s, a, return_dist)

    # remove coordinates, primitive timestep, and room location from the next state
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
        # x_coord, y_coord
        # current primitive timestep
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # human working step: [n_objs + 1]

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
        # x_coord, y_coord
        # current primitive timestep
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # human working step: [n_objs + 1]
        assert len(state) == (2 + 1 + self.n_objs + 2), f"Len state {len(state)} is wrong "

        delta_time = new_state[self.timestep_idx] - state[self.timestep_idx]
        reward = -delta_time

        prev_human_stage = state[-1]

        new_human_stage = new_state[-1]

        # Deliver a good tool
        if prev_human_stage + 1 == new_human_stage and action == self.n_objs:
            reward += 100

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
        # x_coord, y_coord
        # current primitive timestep
        # discrete room locations: [2]
        # which object in the basket: [2]*n_objs
        # human working step: [n_objs + 1]

        done = False
        human_stage = new_state[-1]

        if human_stage == 3:
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

    def sample(self, prior=None, obs_correct_prob=1.0) -> Domain:
        """
        Prior that the correct order will start with the first tool
        """
        if prior:
            correct_tool_order = list(prior)
        else:
            perm = list(permutations([0, 2, 1]))
            correct_tool_order = random.choice(perm)
        return ToolDeliveryV0(list(correct_tool_order), obs_correct_prob)

if __name__ == "__main__":
    env = ToolDeliveryV0(correct_tool_order=[2, 1, 0], render=False)
    env.reset()

    index_2_str = ['Get_Tool_0', 'Get_Tool_1', 'Get_Tool_2', 'Deliver']

    def action_to_str(action_idx):
        return f"Action: {index_2_str[action_idx]}"

    optimal = True

    rewards = []

    if optimal:

        act_list = [GET_TOOL_2] + [DELIVER] + [GET_TOOL_1] + [DELIVER] + [GET_TOOL_0] + [DELIVER]

        for action in act_list:
            print(action_to_str(action))
            state = env.step(action)
            print("Timestep:", env.get_state()[3])
            print(state.observation, state.reward, state.terminal)
            print()
            rewards.append(state.reward)
    else:
        act_list = [GET_TOOL_0, GET_TOOL_1, GET_TOOL_2] + [DELIVER]*5

        for action in act_list:
            print(action_to_str(action))
            state = env.step(action)
            print(state.reward, state.observation, state.terminal)
            print(env.get_state())
            print("Timestep:", env.get_state()[3])
            print()
            rewards.append(state.reward)
            time.sleep(1)

    print(rewards)
    discounted_return = sum(
        pow(0.95, i) * r for i, r in enumerate(rewards)
    )

    print(optimal, discounted_return)
