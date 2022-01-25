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
    tot_counts_for_known = 10000.0
    
XXXX here I am not clear why we set correctness here, but I think it is just the implementation detail that not matter much
    correct_prob = 0.625 + (correctness * 0.225)
    
    # transition counts
    # It is an implementation detail that should not matter much
    # we will design a empty array at first
    # here we have 810 states, 5 actions
    # the shape of t_realdomain is (810, 5, 810)-> 3 dimensions
XXXX here action will change the location of the agent, the status of tools and human working status.
XXXX Please allow me to set an example, if the inital state s is [0,5,2,2,2,2] and the action is go-to-workroom
XXXX Then the state s‘ is [1,5,2,2,2,2]. How can we determine the index value of these two states?
XXXX Or in this situation t_realdomain[?,0,?]

    t_realdomain = np.zeros([810,5,810],dtype=float)


    # observation prior counts
    # 5 actions: go-to-workroom, go-to-toolroom, pickup, deliver, listen
    # 2 locations: workroom and tool room
    # 5 different working stage
    # tool status has 3 values-> 0: work room; 1: with the robot; 2: deliverd
    # 6 observations: 0, 1, 2, 3, 4, nothing
    
    # For the rest part, the listening content depends on human's working status
    a = 1 - correct_prob
    o_realdomain = np.zeros([5,810,6],dtype=float)
    # If the agent execute non-observation actions, then it will observes nothing
    # here action == go-to-workroom || go-to-toolroom || pick-up || drop
    o_realdomain[0:4,0:810,:] = [p * tot_counts_for_known for p in [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    
    # we assume the second half indicates that the Turtlebot located at work room
    # In the first part which indicates that the Turtlebot located at tool room
    # The agent still observes nothing
    o_realdomain[4,0:405,:] = [p * certainty for p in [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    # In the second part, which indicates that the Turtlebot located at workroom
    # Only when the agent in the work room and listen, it can observe human's current stauts:
    # what it will observe based on human's working status
    
    # human_working_stauts == 0 && action == listen && the agent located at workroom
    o_realdomain[4,405:486,:] = [p * certainty for p in [1-a, a/4, a/4, a/4, a/4, 0.0]]
    # human_working_stauts == 1 && action == listen && the agent located at workroom
    o_realdomain[4,486:567,:] = [p * certainty for p in [a/4, 1-a, a/4, a/4, a/4, 0.0]]
    # human_working_stauts == 2 && action == listen && the agent located at workroom
    o_realdomain[4,567:648,:] = [p * certainty for p in [a/4, a/4, 1-a, a/4, a/4, 0.0]]
    # human_working_stauts == 3 && action == listen && the agent located at workroom
    o_realdomain[4,648:729,:] = [p * certainty for p in [a/4, a/4, a/4, 1-a, a/4, 0.0]]
    # human_working_stauts == 4 && action == listen && the agent located at workroom
    o_realdomain[4,729:810,:] = [p * certainty for p in [a/4, a/4, a/4, a/4, 1-a, 0.0]]
    

    return DirCounts(t_realdomain, o_realdomain)