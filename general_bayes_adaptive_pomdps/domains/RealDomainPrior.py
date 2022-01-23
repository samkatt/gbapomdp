"""Here is our problem designed as RealDomain Problem"""
from logging import Logger
from typing import List, Optional

import numpy as np

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
)
from general_bayes_adaptive_pomdps.domains.domain import Domain, DomainPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace, LogLevel
from general_bayes_adaptive_pomdps.models.tabular_bapomdp import DirCounts




class RealDomainPrior(DomainPrior):
    """standard prior over the tiger domain
    The transition model is known, however the probability of observing the
    tiger correctly is not. Here we assume a `Dir(prior * total_counts
    ,(1-prior) * total_counts)` belief over this distribution.
    ``prior`` is computed by the ``prior_correctness``: 1 -> .85, whereas 0 ->
    .625, linear mapping in between
    """

    def __init__(
        self,
        num_total_counts: float,
        prior_correctness: float,
        one_hot_encode_observation: bool,
    ):
        """initiate the prior, will make observation one-hot encoded
        Args:
             num_total_counts: (`float`): Number of total counts of Dir prior
             prior_correctness: (`float`): How correct the observation model is: [0, 1] -> [.625, .85]
             one_hot_encode_observation: (`bool`):
        """
        super().__init__()

        if num_total_counts <= 0:
            raise ValueError(
                f"Assume positive number of total counts, not {num_total_counts}"
            )

        if not 0 <= prior_correctness <= 1:
            raise ValueError(
                f"`prior_correctness` must be [0,1], not {prior_correctness}"
            )
        # prior_correctness = []
        # Linear mapping: [0, 1] -> [.625, .85]
        self._observation_prob = 0.625 + (prior_correctness * 0.225)
        self._total_counts = num_total_counts
        self._one_hot_encode_observation = one_hot_encode_observation

    def sample(self) -> Domain:
        """returns a Tiger instance with some correct observation prob
        This prior over the observation probability is a Dirichlet with total
        counts and observation probability as defined during the initialization
        RETURNS (`general_bayes_adaptive_pomdps.core.Domain`):
        """
        # to get the probabilities of correct observation based on different human working status
        # while the agent in the workroom.
        sampled_observation_probs = [
            np.random.dirichlet(
                [
                    self._observation_prob * self._total_counts,
                    (1 - self._observation_prob) * self._total_counts,
                ]
            )[0],
            np.random.dirichlet(
                [
                    self._observation_prob * self._total_counts,
                    (1 - self._observation_prob) * self._total_counts,
                ]
            )[0],
            np.random.dirichlet(
                [
                    self._observation_prob * self._total_counts,
                    (1 - self._observation_prob) * self._total_counts,
                ]
            )[0],
            np.random.dirichlet(
                [
                    self._observation_prob * self._total_counts,
                    (1 - self._observation_prob) * self._total_counts,
                ]
            )[0],
            np.random.dirichlet(
                [
                    self._observation_prob * self._total_counts,
                    (1 - self._observation_prob) * self._total_counts,
                ]
            )[0],
        ]
        # 之后会设定参数
        return RealDomain(
            one_hot_encode_observation=self._one_hot_encode_observation,
            correct_obs_probs=sampled_observation_probs,
        )

# 关于设定 prior counts的概率
def create_tabular_prior_counts(
    correctness: float = 1, certainty: float = 10
) -> DirCounts:
    """Creates a prior of :class:`Realdomain` for the :class:`TabularBAPOMDP`
    The prior for this problem is "correct" and certain about the transition
    function, but the correctness and certainty over the observation model
    (when listening) is variable. `correctness` provides a way to linearly
    interpolate between a prior that assigns (0 => 0.625 ... 1 => 0.85)
    probability to generating the correct observation. The `certainty`
    describes how many (total) counts this prior should have.
    The prior used in most BRL experiments is where `correctness = 0` and
    `certainty = 8`.
    # 0 为最不正确, 1为正确，一般默认为1
    # 返回 DirCounts也就是一系列的先验
    Args:
        correctness: (`float`): 0 for "most incorrect", 1 for "correct", default 1
        certainty: (`float`): total number of counts in observation prior
    RETURNS (`DirCounts`): a (set of dirichlet counts) prior
    """
    # certainty 为observation先验计数
    # 总计数
    # episode
    # horizon
    # certainty 用法
    tot_counts_for_known = 10000.0
    correct_prob = 0.625 + (correctness * 0.225)
    # transition counts
    # t also involves state and action
    # In the original tiger problem, the agent may hear either left or right with 50%.
    # It is an implementation detail that should not matter much
    # we still design a empty array at first
    # in the original array, its shape is [2,3,2] -> 2 states, 3 actions 2 observations;
    # here we have 2*5*3*3*3*3 states, 5 actions, 5 observations
    t_realdomain = np.zeros([2,5,3,3,3,3,5,5],dtype=float)
    # for the action other than listening, we set the same probabilities for each observation
    # here we can set a value formulation in the future.
    t_realdomain[0:2,0:5,0:3,0:3,0:3,0:3,0:4,:] = [p * tot_counts_for_known for p in [0.2, 0.2, 0.2, 0.2, 0.2]]
    # here only involves listening action with different human status
    t_realdomain[0:2,0,0:3,0:3,0:3,0:3,4,:] = [p * tot_counts_for_known for p in [1.0, 0.0, 0.0, 0.0, 0.0]]
    t_realdomain[0:2,1,0:3,0:3,0:3,0:3,4,:] = [p * tot_counts_for_known for p in [0.0, 1.0, 0.0, 0.0, 0.0]]
    t_realdomain[0:2,2,0:3,0:3,0:3,0:3,4,:] = [p * tot_counts_for_known for p in [0.0, 0.0, 1.0, 0.0, 0.0]]
    t_realdomain[0:2,3,0:3,0:3,0:3,0:3,4,:] = [p * tot_counts_for_known for p in [0.0, 0.0, 0.0, 1.0, 0.0]]
    t_realdomain[0:2,4,0:3,0:3,0:3,0:3,4,:] = [p * tot_counts_for_known for p in [0.0, 0.0, 0.0, 0.0, 1.0]]

    # observation prior counts
    # 5 actions: go-to-workroom, go-to-toolroom, pickup, deliver, listen
    # 2 locations: workroom and tool room
    # 5 different working stage
    # 4 tools with 3 values-> 0: work room; 1: with the robot; 2: deliverd
    # 6 observations: 0 1 2 3 4 nothing
    # since it is of large dimension, we need to create empty array at first;
    o_realdomain = np.zeros([5,2,5,3,3,3,3,6],dtype=float)
    # a = listen and location in work room and based on human working status, only under this condition can the valid produce;
    # first of all, we can set the rest parts as seeing nothing!
    o_realdomain[0:4,0:2,0:5,0:3,0:3,0:3,0:3,:] = [p * tot_counts_for_known for p in [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    # listening part:
    # The listening part in the workroom listen nothing as well.
    o_realdomain[4,0,0:5,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    # For the rest part, the listening content depends on human's working status
    a = 1 - correct_prob
    o_realdomain[4,1,0,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [1-a, a/4, a/4, a/4, a/4, 0.0]]
    o_realdomain[4,1,1,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [a/4, 1-a, a/4, a/4, a/4, 0.0]]
    o_realdomain[4,1,2,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [a/4, a/4, 1-a, a/4, a/4, 0.0]]
    o_realdomain[4,1,3,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [a/4, a/4, a/4, 1-a, a/4, 0.0]]
    o_realdomain[4,1,4,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [a/4, a/4, a/4, a/4, 1-a, 0.0]]
    
    # Since the agent knows its tool status for the observation, I set another possibilities:
    # o_realdomain[4,1,0,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [1-a, a/4, a/4, a/4, a/4, 0.0]]
    # o_realdomain[4,1,1,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [0.0, 1-a, a/3, a/3, a/3, 0.0]]
    # o_realdomain[4,1,2,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [0.0, 0.0, 1-a, a/2, a/2, 0.0]]
    # o_realdomain[4,1,3,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [0.0, 0.0, 0.0, 1-a, a, 0.0]]
    # o_realdomain[4,1,4,0:3,0:3,0:3,0:3,:] = [p * certainty for p in [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]

    return DirCounts(t_realdomain, o_realdomain)