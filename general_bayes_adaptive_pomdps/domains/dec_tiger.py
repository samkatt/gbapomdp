"""The tiger problem implemented as domain"""

from logging import Logger
from typing import List, Optional

import numpy as np

import one_to_one

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace, LogLevel
from general_bayes_adaptive_pomdps.models.tabular_bapomdp import DirCounts


class DecTiger(Domain):
    """The actual domain"""

    # Observation (Hear-Left, Hear-Right)
    O_HL = 0
    O_HR = 1

    # State (Tiger-Left, Tiger-Right)
    S_TL = 0
    S_TR = 1

    # Action (Open-Left, Open-Right, Listen)
    A_OL = 0
    A_OR = 1
    A_LI = 2

    def __init__(
        self,
        one_hot_encode_observation: bool,
        correct_obs_probs: Optional[List[float]] = None,
    ):
        """Construct the multiger domain

        Args:
             one_hot_encode_observation: (`bool`):
             correct_obs_probs: (`Optional[List[float]]`):

        """
        super().__init__()

        if not correct_obs_probs:
            correct_obs_probs = [0.85, 0.85]

        assert (
            0 <= correct_obs_probs[0] <= 1
        ), f"observation prob {correct_obs_probs[0]} not a probability"
        assert (
            0 <= correct_obs_probs[1] <= 1
        ), f"observation prob {correct_obs_probs[1]} not a probability"

        self._correct_obs_probs = correct_obs_probs

        self._use_one_hot_obs = one_hot_encode_observation

        # (S_TL, S_TR)
        self.sem_state_space = one_to_one.JointNamedSpace(
            state=one_to_one.RangeSpace(2),
        )
        self._state_lst = list(self.sem_state_space.elems)
        self._state_space = DiscreteSpace([self.sem_state_space.nelems])

        # (O_HL, O_HR) x (O_HL, O_HR)
        self.sem_obs_space = one_to_one.JointNamedSpace(
            agent_0=one_to_one.RangeSpace(2),
            agent_1=one_to_one.RangeSpace(2),
        )
        self._obs_lst = list(self.sem_obs_space.elems)
        self._obs_space = (
            DiscreteSpace([self.sem_obs_space.nelems])
        )

        # (A_OL, A_OR, A_LI) x (A_OL, A_OR, A_LI)
        self.sem_act_space = one_to_one.JointNamedSpace(
            agent_0=one_to_one.RangeSpace(3),
            agent_1=one_to_one.RangeSpace(3),
        )
        self._act_lst = list(self.sem_act_space.elems)
        self._action_space = ActionSpace(self.sem_act_space.nelems)

        self._state = self.sample_start_state()

    @property
    def state(self):
        """returns current state"""
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        """sets state

        Args:
             state: (`np.ndarray`):

        """

        self._state = state

    @property
    def state_space(self) -> DiscreteSpace:
        """a `general_bayes_adaptive_pomdps.misc.DiscreteSpace` ([2]) space"""
        return self._state_space

    @property
    def action_space(self) -> ActionSpace:
        """a `general_bayes_adaptive_pomdps.core.ActionSpace` ([3]) space"""
        return self._action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """a `general_bayes_adaptive_pomdps.misc.DiscreteSpace` ([1,1]) space if one-hot, otherwise [3]"""
        return self._obs_space

    @staticmethod
    def sample_start_state() -> np.ndarray:
        """samples a random state (tiger left or right)

        RETURNS (`np.narray`): an initial state (in [[0],[1]])

        """
        return np.array([np.random.randint(0, 2)], dtype=int)

    def reset(self) -> np.ndarray:
        """Resets internal state and return first observation

        Resets the internal state randomly ([0] or [1])
        Returns [1,1] as a 'null' initial observation

        """
        self._state = self.sample_start_state()

        return self._prepare_obs(self._state, 0, reset=True)

    def _prepare_obs(self, n_state, action, o_prob=0.85, reset=False) -> np.ndarray:

        curr_action = self._act_lst[action]
        temp_obs = self._obs_lst[0]

        if reset:
            return temp_obs

        if (curr_action.agent_0.value != self.A_LI or curr_action.agent_1.value != self.A_LI):
            temp_obs.agent_0.value = np.random.choice((self.O_HL, self.O_HR))
            temp_obs.agent_1.value = np.random.choice((self.O_HL, self.O_HR))
        else:
            next_state = self._state_lst[n_state[0]]

            if next_state.state.value == self.S_TL:
                temp_obs.agent_0.value = self.O_HL if np.random.rand() <= o_prob else self.O_HR
                temp_obs.agent_1.value = self.O_HL if np.random.rand() <= o_prob else self.O_HR
            elif next_state.state.value == self.S_TR:
                temp_obs.agent_0.value = self.O_HR if np.random.rand() <= o_prob else self.O_HL
                temp_obs.agent_1.value = self.O_HR if np.random.rand() <= o_prob else self.O_HL
            else:
                print("Invalid case")

        return temp_obs

    def simulation_step(self, state: np.ndarray, action: int, d_prob=0.5, o_prob=0.85) -> SimulationResult:
        """Simulates stepping from state using action. Returns interaction

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             state: (`np.ndarray`):
             action: (`int`):

        RETURNS (`general_bayes_adaptive_pomdps.core.SimulationResult`): the transition

        """

        curr_action = self._act_lst[action]

        # At least one agent opens door, next state can be S_TL or S_TR with equal prob.
        if (curr_action.agent_0.value != self.A_LI or curr_action.agent_1.value != self.A_LI):
            new_state = np.array([self.S_TL] if np.random.rand() <= d_prob else [self.S_TR])
        # If both listen, the state does not change
        else:
            new_state = np.array([state[0]])

        obs = self._prepare_obs(new_state, action, o_prob)

        return SimulationResult(new_state, np.array([obs.idx]))

    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """A constant if listening, penalty if opening to door, and reward otherwise

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`float`):

        """

        assert self.state_space.contains(state), f"{state} not in space"
        assert self.state_space.contains(new_state), f"{new_state} not in space"
        assert self.action_space.contains(action), f"{action} not in space"

        curr_action = self._act_lst[action]
        curr_state = self._state_lst[state[0]]

        if (curr_action.agent_0.value, curr_action.agent_1.value) == (self.A_OR, self.A_OR):
            if curr_state.state.value == self.S_TL:
                return 20
            else:
                return -50

        if (curr_action.agent_0.value, curr_action.agent_1.value) == (self.A_OL, self.A_OL):
            if curr_state.state.value == self.S_TL:
                return -50
            else:
                return 20

        if (curr_action.agent_0.value, curr_action.agent_1.value) == (self.A_OR, self.A_OL):
            return -100

        if (curr_action.agent_0.value, curr_action.agent_1.value) == (self.A_OL, self.A_OR):
            return -100

        if (curr_action.agent_0.value, curr_action.agent_1.value) == (self.A_LI, self.A_LI):
            return -2

        if (curr_action.agent_0.value, curr_action.agent_1.value) == (self.A_LI, self.A_OR):
            if curr_state.state.value == self.S_TL:
                return 9
            else:
                return -101

        if (curr_action.agent_0.value, curr_action.agent_1.value) == (self.A_OR, self.A_LI):
            if curr_state.state.value == self.S_TL:
                return 9
            else:
                return -101

        if (curr_action.agent_0.value, curr_action.agent_1.value) == (self.A_LI, self.A_OL):
            if curr_state.state.value == self.S_TL:
                return -101
            else:
                return 9

        if (curr_action.agent_0.value, curr_action.agent_1.value) == (self.A_OL, self.A_LI):
            if curr_state.state.value == self.S_TL:
                return -101
            else:
                return 9

        print("Should never reach here")

    def get_state(self):
        return self.state

    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if opening a door

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):

        RETURNS (`bool`):

        """

        assert self.state_space.contains(state), f"{state} not in space"
        assert self.state_space.contains(new_state), f"{new_state} not in space"
        assert self.action_space.contains(action), f"{action} not in space"

        curr_action = self._act_lst[action]

        return bool(curr_action.agent_0.value != self.A_LI or curr_action.agent_1.value != self.A_LI)

    def step(self, action: int) -> DomainStepResult:
        """Performs a step in the tiger problem given action

        Will terminate episode when action is to open door,
        otherwise return an observation.

        Args:
             action: (`int`): 0 is open left, 1 is open right or 2 is listen

        RETURNS (`general_bayes_adaptive_pomdps.core.EnvironmentInteraction`): the transition

        """

        sim_result = self.simulation_step(self.state, action)
        reward = self.reward(self.state, action, sim_result.state)
        terminal = self.terminal(self.state, action, sim_result.state)

        self.state = sim_result.state

        return DomainStepResult(sim_result.observation, reward, terminal)


def create_tabular_prior_counts(
    correctness: float = 1, certainty: float = 10
) -> DirCounts:
    """Creates a prior of :class:`Tiger` for the :class:`TabularBAPOMDP`

    The prior for this problem is "correct" and certain about the transition
    function, but the correctness and certainty over the observation model
    (when listening) is variable. `correctness` provides a way to linearly
    interpolate between a prior that assigns (0 => 0.625 ... 1 => 0.85)
    probability to generating the correct observation. The `certainty`
    describes how many (total) counts this prior should have.

    The prior used in most BRL experiments is where `correctness = 0` and
    `certainty = 8`.

    Args:
        correctness: (`float`): 0 for "most incorrect", 1 for "correct", default 1
        certainty: (`float`): total number of counts in observation prior

    RETURNS (`DirCounts`): a (set of dirichlet counts) prior

    """

    tot_counts_for_known = 1000.0

    env = DecTiger(False)

    state_space = env.sem_state_space
    obs_space = env.sem_obs_space
    action_space = env.sem_act_space

    T_mat = np.zeros((state_space.nelems,  action_space.nelems, state_space.nelems))
    O_mat = np.zeros((action_space.nelems, state_space.nelems,  obs_space.nelems))

    for s in state_space.elems:
        for a in action_space.elems:
            for next_s in state_space.elems:
                if (a.agent_0.value != env.A_LI or a.agent_1.value != env.A_LI):
                    T_mat[s.idx, a.idx, next_s.idx] = 0.5
                else:
                    if s.state.value == next_s.state.value:
                        T_mat[s.idx, a.idx, next_s.idx] = 1.0

    for a in action_space.elems:
        for s in state_space.elems:
            for o in obs_space.elems:
                if (a.agent_0.value != env.A_LI or a.agent_1.value != env.A_LI):
                    O_mat[a.idx, s.idx, o.idx] = 0.5*0.5
                else:
                    if s.state.value == env.S_TL:
                        if o.agent_0.value == env.O_HL and o.agent_1.value == env.O_HL:
                            O_mat[a.idx, s.idx, o.idx] = 0.85*0.85

                        if o.agent_0.value == env.O_HL and o.agent_1.value == env.O_HR:
                            O_mat[a.idx, s.idx, o.idx] = 0.85*0.15

                        if o.agent_0.value == env.O_HR and o.agent_1.value == env.O_HL:
                            O_mat[a.idx, s.idx, o.idx] = 0.15*0.85

                        if o.agent_0.value == env.O_HR and o.agent_1.value == env.O_HR:
                            O_mat[a.idx, s.idx, o.idx] = 0.15*0.15
                    else:
                        if o.agent_0.value == env.O_HL and o.agent_1.value == env.O_HL:
                            O_mat[a.idx, s.idx, o.idx] = 0.15*0.15

                        if o.agent_0.value == env.O_HL and o.agent_1.value == env.O_HR:
                            O_mat[a.idx, s.idx, o.idx] = 0.15*0.85

                        if o.agent_0.value == env.O_HR and o.agent_1.value == env.O_HL:
                            O_mat[a.idx, s.idx, o.idx] = 0.85*0.15

                        if o.agent_0.value == env.O_HR and o.agent_1.value == env.O_HR:
                            O_mat[a.idx, s.idx, o.idx] = 0.85*0.85

    return DirCounts(T_mat*tot_counts_for_known, O_mat*tot_counts_for_known)
