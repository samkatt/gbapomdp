"""The tabular BA-POMDP with Dirichlet priors

This is the GBA-POMDP that represents a dynamics model posterior as a table of
Dirichlet distributions. The Bayesian estimate of the transition model, for
example, is a `|S|` sized Dirichlet distribution for each state-action pair. To
update a set of Dirichlets given a new transition simply increment the count of
the associated transition.

"""
from copy import deepcopy
from typing import NamedTuple, NewType

import numpy as np

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStatePrior,
    GeneralBAPOMDP,
    RewardFunction,
    SimulationResult,
    TerminalFunction,
)
from general_bayes_adaptive_pomdps.misc import DiscreteSpace


def sample_large_dirichlet(counts: np.ndarray) -> int:
    """Samples from the (large) distribution induced by the dirichlets `counts`

    A Dirichlet distirbution is a distribution over a categorical distribution
    over values 0 ... `len(counts)` - 1. Conceptually, counts[i] represents how
    often we have observed value i.

    NOTE: here we sample directly from the expected model, we do not sample
    categorical distributions.

    Faster than `sample_small_dirichlet` when `len(counts) > 75`

    :param counts: a 1-dimnesional numpy array
    :returns: a value between 0 and len(counts)
    """
    return np.random.choice(a=len(counts), p=counts / counts.sum())


def sample_small_dirichlet(counts: np.ndarray) -> int:
    """Samples from the distribution induced by the dirichlets `counts`

    A Dirichlet distirbution is a distribution over a categorical distribution
    over values 0 ... `len(counts)` - 1. Conceptually, counts[i] represents how
    often we have observed value i.

    NOTE: here we sample directly from the expected model, we do not sample
    categorical distributions.

    Faster than `sample_large_dirichlet` when `len(counts) < 75`

    :param counts: a 1-dimnesional numpy array
    :returns: a value between 0 and len(counts)
    """
    tot = np.sum(counts)
    p = np.random.random() * tot

    acc = 0.0

    for i, c in enumerate(counts):
        acc += c
        if p < acc:
            return i

    raise ValueError(f"sampled 'probability' {p} < {tot} not reached with {counts}")


TransitionCounts = NewType("TransitionCounts", np.ndarray)
"""Bayesian tabular estimate of the transition model (3-D)"""

ObservationCounts = NewType("ObservationCounts", np.ndarray)
"""Bayesian tabular estimate of the observation model (3-D)"""


class DirCounts(NamedTuple):
    """A tabular Bayesian estimate of the POMDP model"""

    T: TransitionCounts
    O: ObservationCounts


class TBAPOMDPState(NamedTuple):
    """A (hyper) state in the tabular BA-POMDP

    For the fun of it I decided to attempt a more functional interface. In
    stead of implementing functionality as members of this class, I have
    defined stand-alone functions that take this class as input
    """

    domain_state: np.ndarray
    counts: DirCounts


class TabularBAPOMDP(GeneralBAPOMDP[TBAPOMDPState]):
    """The tabular Dirichlet GBA-POMDP

    Implements the GeneralBAPOMDP interface
    """

    def __init__(
        self,
        state_space: DiscreteSpace,
        action_space: ActionSpace,
        observation_space: DiscreteSpace,
        sample_domain_start_state: DomainStatePrior,
        reward_function: RewardFunction,
        terminal_function: TerminalFunction,
        model_prior: DirCounts,
    ):
        """Creates a (GBA-) POMDP from the prior `model_prior`"""
        # domain knowledge
        self.domain_state_space = state_space
        self.domain_action_space = action_space
        self.domain_obs_space = observation_space

        self.sample_domain_start_state = sample_domain_start_state
        self.domain_reward = reward_function
        self.domain_terminal = terminal_function

        self.model_prior = model_prior

        self.sample_t = (
            sample_small_dirichlet if state_space.ndim < 50 else sample_large_dirichlet
        )
        self.sample_o = (
            sample_small_dirichlet
            if observation_space.ndim < 50
            else sample_large_dirichlet
        )

    @property
    def action_space(self) -> ActionSpace:
        """The action space of the underlying POMDP"""
        return self.domain_action_space

    @property
    def observation_space(self) -> DiscreteSpace:
        """The observation space of the underlying POMDP"""
        return self.domain_obs_space

    def sample_start_state(self) -> TBAPOMDPState:
        """samples an initial `TBAPOMDPState` in the tabular GBA-POMDP

        Returns a domain state sampled from the given
        `sample_domain_start_state` and pairs it with the `model_prior`

        :returns: an initial `AugmentedState` in this (GBA-)POMDP
        """
        return TBAPOMDPState(
            self.sample_domain_start_state(),
            deepcopy(self.model_prior),
        )

    def domain_simulation_step(
        self, state: TBAPOMDPState, action: int
    ) -> SimulationResult:
        """Samples a next state and observation

        Samples according to the counts in `state`, given that the current
        state is the domain state in `state` and the agent took `action`.

        NOTE: that the model in the returned state is a reference to the one in
        `state` so be careful: if you modify either, you modify both

        :param state: (augmented) state (counts and domain state) at timestep t
        :param action: action taken at timestep t
        :returns: (augmented state, obs) at t+1, where model in state == of the incoming `state`
        """

        state_index = self.domain_state_space.index_of(state.domain_state)

        next_state_index = self.sample_t(state.counts.T[state_index, action])
        observation_index = self.sample_o(state.counts.O[action, next_state_index])

        next_state = self.domain_state_space.from_index(next_state_index)
        observation = self.observation_space.from_index(observation_index)

        return SimulationResult(TBAPOMDPState(next_state, state.counts), observation)

    def model_simulation_step(
        self,
        to_update: TBAPOMDPState,
        prev_state: TBAPOMDPState,
        action: int,
        next_state: TBAPOMDPState,
        obs: np.ndarray,
        optimize: bool = False,
    ) -> TBAPOMDPState:
        """Increments the counts in `to_update` associated with the (s, a, s', o) transition

        NOTE: the domain state in the returned state is a reference to the one
        in `to_update` so be careful: if you modify either, you modify both

        Note that this operation is expensive, as it involves generating a new
        set of counts to represent the model distribution in the new state
        (which involves a copy before updating). If there is no need for the
        input `to_update` to stay unmodified, set `optimize` to `True`, and the
        copy is avoided (but the incoming `to_update` is modified!)

        :param to_update: the augmented state that contains the model at timestep t
        :param prev_state: the state at timestep t of the transition
        :param action: the action at timestep t of the transition
        :param next_state: the state at timestep t+1 of the transition
        :param obs: the observation at timestep t of the transition
        :param optimize: optimization flag.
            If set to true, the model in `to_update` is _not_ copied, and
            thus the incoming `to_update` **is modified**. If there is no
            need to keep the old model, then setting this flag skips a then
            needless copy operation, which can be significant
        :returns: new state with updated model (same domain state as `to_update`)
        """

        index_s = self.domain_state_space.index_of(prev_state.domain_state)
        index_next_s = self.domain_state_space.index_of(next_state.domain_state)
        index_o = self.observation_space.index_of(obs)

        updated_counts = to_update.counts if optimize else deepcopy(to_update.counts)

        updated_counts.T[index_s, action, index_next_s] += 1
        updated_counts.O[action, index_next_s, index_o] += 1

        return TBAPOMDPState(to_update.domain_state, updated_counts)

    def simulation_step(
        self, state: TBAPOMDPState, action: int, optimize: bool = False
    ) -> SimulationResult:
        """Performs an actual step according to the GBA-POMDP dynamics

        The resulting `SimulationResult` contains the generated state and
        observation, where the state is generated by sampling according to the
        counts in `state`.

            1. sample a new domain state and observation according to the counts
               and current domain state in `state`
            2. update counts in `state` that is associated with sampled transition

        Note that this operation is expensive, as it often involves generating
        a new set of parameters to represent the model distribution in the new
        state (which involves a copy before updating). If there is no need for
        the input `state` to stay unmodified, set `optimize` to `True`

        :param state: state at timestep t
        :param action: action at timestep t
        :param optimize: optimization flag.
            If set to true, the model in `state` is _not_ copied, and thus
            the incoming `state` **is modified**. If there is no need to keep
            the old model, then setting this flag skips a then needless copy
            operation, which can be significant
        :return: (state, obs) at timestep t+1
        """

        # sample transition
        index_s = self.domain_state_space.index_of(state.domain_state)

        index_next_s = self.sample_t(state.counts.T[index_s, action])
        index_o = self.sample_o(state.counts.O[action, index_next_s])

        updated_counts = state.counts if optimize else deepcopy(state.counts)

        # update counts
        updated_counts.T[index_s, action, index_next_s] += 1
        updated_counts.O[action, index_next_s, index_o] += 1

        # re-index into features
        observation = self.observation_space.from_index(index_o)
        next_state = self.domain_state_space.from_index(index_next_s)

        return SimulationResult(TBAPOMDPState(next_state, updated_counts), observation)

    def reward(
        self, state: TBAPOMDPState, action: int, next_state: TBAPOMDPState
    ) -> float:
        """the reward function of the underlying domain"""
        return self.domain_reward(state.domain_state, action, next_state.domain_state)

    def terminal(
        self, state: TBAPOMDPState, action: int, next_state: TBAPOMDPState
    ) -> bool:
        """the termination function of the underlying domain"""
        return self.domain_terminal(state.domain_state, action, next_state.domain_state)
