""" Particle Observable UCT """
import random
from collections import namedtuple
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
from po_nrl.agents.planning.particle_filters import ParticleFilter
from po_nrl.environments import ActionSpace, Simulator
from po_nrl.misc import POBNRLogger
from typing_extensions import Protocol  # pylint: disable=wrong-import-order

_IterationResult = namedtuple('IterResult', 'ret depth length')


class TreeNode:
    """node in the tree

    MTCS in POMDPs branches both in actions and states, which (naively) would
    result in two types of nodes. However, in practice its easier to represent
    both branches in a single node:

    A node in this MCTS tree represents a particular 'future history' and thus
    is associated with an action-observation history. Its children refers to
    the next action-observations. Technically, it describes a _set_ of nodes,
    each associated with an action.
    """

    def __init__(self, num_children: int, depth: int):
        """ creates a node with `num_children` childs at some depth `depth`

        Args:
             num_children: (`int`): the number of child nodes to have
             depth: (`int`): the depth of `this` in the tree

        """

        self.depth = depth

        # we expect MCTS to consider _all_ actions, so immediately initialize them as list
        # however, there is no way of telling which observations, which could
        # be many, are going to be generated, so their mapping is represented
        # with a dictionary. Its entries are generated on the fly (see
        # `child()`). Observations are encoded into tuples
        self._children: List[Dict[Tuple[int], TreeNode]] = [
            {} for _ in range(num_children)
        ]

        # statistics
        self.avg_values = np.zeros(num_children)
        self.children_visits = np.zeros(num_children).astype(int)

    @property
    def num_visits(self) -> int:
        """ returns number of visits of this

        Args:

        RETURNS (`int`):

        """
        return self.children_visits.sum()

    @property
    def num_children(self) -> int:
        """ returns the number of child nodes

        Args:

        RETURNS (`int`):

        """

        # probably faster than len() over a list?
        return self.avg_values.shape[0]

    def child(self, action: int, observation: np.ndarray) -> 'TreeNode':
        """ returns the child node associated with action-observation pair

        Args:
             action: (`int`): the chosen action
             observation: (`np.ndarray`): the perceived observation as array

        RETURNS (`po_nrl.agents.planning.pouct.TreeNode`):

        """
        obs: Tuple[int] = tuple(observation.flatten().astype(int))  # type: ignore

        # initialize child if observation has not been seen before with this
        # particular action
        if obs not in self._children[action]:
            self._children[action][obs] = TreeNode(
                len(self._children), depth=self.depth + 1
            )

        return self._children[action][obs]

    def update_value(self, child: int, value: float):
        """ updates the value `value` associated with child `child`

        Args:
             child: (`int`): the child to update
             value: (`float`): a new value to be added to the average

        """

        self.children_visits[child] += 1
        self.avg_values[child] += (
            value - self.avg_values[child]
        ) / self.children_visits[child]

    def __repr__(self) -> str:
        """ prints out value and counts for each child """
        # [A(i): v (c) .... ]
        return str(
            [
                f'A({i}): {v:.2f} ({c})'
                for i, (c, v) in enumerate(
                    zip(self.children_visits, self.avg_values)
                )
            ]
        )


class RolloutPolicy(Protocol):
    """interface for rollout policies: state -> action"""

    def __call__(self, state: np.ndarray) -> int:
        """returns an action given `state`

        Args:
            state (`np.ndarray`):

        Returns:
            int: action of the policy
        """


def random_policy(
    state: np.ndarray,  # pylint: disable=unused-argument
    action_space: ActionSpace,
) -> int:
    """random policy selects a random action from the action space

    Args:
        _ (`np.ndarray`): state, ignored, part of `RolloutPolicy` interface
        action_space (`ActionSpace`):

    Returns:
        int: action
    """
    return action_space.sample()


class POUCT(POBNRLogger):
    """ MCTS for POMDPs using UCB """

    def __init__(
        self,
        simulator: Simulator,
        rollout_policy: RolloutPolicy,
        num_sims: int = 500,
        exploration_constant: float = 1.0,
        planning_horizon: int = 10,
        discount: float = 0.95,
    ):
        """ Creates the PO-UCT planner

        Args:
             simulator: (`po_nrl.environments.Simulator`)
             num_sims: (`int`): number of iterations
             exploration_constant: (`float`): UCB exploration constant
             planning_horizon: (`int`): the horizon to plan agains
             discount: (`float`): the discount factor of the env

        """

        assert (
            planning_horizon > 0
        ), f"Cannot accept a negative planning horizon {planning_horizon}"

        POBNRLogger.__init__(self)

        self.num_sims = num_sims
        self.planning_horizon = planning_horizon
        self.discount = discount
        self.simulator = simulator

        if not rollout_policy:
            rollout_policy = partial(
                random_policy, action_space=self.simulator.action_space
            )

        self.rollout_policy = rollout_policy

        tot_visits = action_visits = np.arange(self.num_sims).reshape(1, -1)

        """
        precompute ucb table. table is such that:
        [m,n] = ucb for m total visits and n action visits
        rows: [[tot_visit=0],[tot_visit=2]...]
        where colums = [act_visit=0, act_visit=1 ...]
        """

        # + 1 for UCB1 (otherwise after trying 1 action all bonus == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            self._ucb_table = exploration_constant * np.sqrt(
                np.log(tot_visits + 1).T / action_visits
            )

    def select_action(self, belief: ParticleFilter) -> int:
        """ selects an action given belief

        Args:
             belief: (`po_nrl.agents.planning.particle_filters.ParticleFilter`): the belief at the root to plan from

        RETURNS (`int`):

        """

        root = TreeNode(self.simulator.action_space.n, depth=0)

        # diagnostics
        min_return = float('inf')
        max_return = -float('inf')
        tree_depth = 0
        longest_iteration = 0

        # build tree
        for run in range(self.num_sims):
            result = self._traverse_tree(belief.sample(), root)

            self.log(
                POBNRLogger.LogLevel.V4, f"POUCT iteration {run}: {result}"
            )

            min_return = min(min_return, result.ret)
            max_return = max(max_return, result.ret)
            tree_depth = max(tree_depth, result.depth)
            longest_iteration = max(longest_iteration, result.length)

        self.log(
            POBNRLogger.LogLevel.V3,
            f"POUCT: Q: {root}, returns {min_return} to {max_return} "
            f"tree depth {tree_depth}, longest run {longest_iteration}",
        )

        # pick best action from root
        return np.argmax(root.avg_values)

    def _traverse_tree(
        self, state: np.ndarray, node: TreeNode
    ) -> _IterationResult:
        """ Travels down the tree

        Picks actions according to UCB, generates transitions according to
        simulator

        Args:
             state: (`np.ndarray`): current state
             node: (`TreeNode`): the current node of the tree

        RETURNS (`_IterationResult`): return of this traversal

        """

        if node.depth == self.planning_horizon:
            return _IterationResult(ret=0, depth=node.depth, length=node.depth)

        if node.num_visits == 0:
            action, ret, length = self._rollout(
                state, self.planning_horizon - node.depth
            )
            iteration_res = _IterationResult(
                ret=ret, depth=node.depth, length=length + node.depth
            )

        else:

            ucbs = POUCT.ucb(
                node.avg_values, node.children_visits, self._ucb_table
            )
            action = random.choice(np.argwhere(ucbs == ucbs.max()).flatten())

            step = self.simulator.simulation_step(state, action)

            reward = self.simulator.reward(state, action, step.state)
            terminal = self.simulator.terminal(state, action, step.state)

            if self.log_is_on(POBNRLogger.LogLevel.V5):
                self.log(
                    POBNRLogger.LogLevel.V5,
                    f"MCTS simulated action {action} in {state} -->"
                    f" {step.state} and obs {step.observation}",
                )

            if not terminal:

                iteration_res = self._traverse_tree(
                    step.state, node.child(action, step.observation),
                )

                iteration_res = iteration_res._replace(
                    ret=reward + self.discount * iteration_res.ret
                )

            else:
                iteration_res = _IterationResult(
                    ret=reward, depth=node.depth, length=node.depth
                )

        node.update_value(action, iteration_res.ret)

        return iteration_res

    def _rollout(self, state: np.ndarray, hor: int) -> Tuple[int, float, int]:
        """ A random interaction with the simulator

        Args:
             state: (`np.ndarray`): state to rollout from
             hor: (`int`): the length of the rollout

        RETURNS (`Tuple[int, float, int]`): the inital action, (discounted) return of the rollout, and length of rollout

        """

        ret = 0.0
        discount = 1.0

        # save this as output
        first_action = self.rollout_policy(state)
        action = first_action

        for time_step in range(hor):

            step = self.simulator.simulation_step(state, action)
            reward = self.simulator.reward(state, action, step.state)
            terminal = self.simulator.terminal(state, action, step.state)

            ret += discount * reward

            discount *= self.discount

            if terminal:
                break

            action = self.rollout_policy(step.state)
            state = step.state

        return (first_action, ret, time_step)

    @staticmethod
    def ucb(
        action_values: np.array,  # of type floats
        action_visits: np.array,  # of type int
        ucb_table: np.ndarray,
    ) -> np.array:  # of type floats
        """ Returns the UCB value given Qs, visits and ucb table

        Args:
             action_values: (`np.array`): list of q values (foreach action)
             action_visits: (`np.array`):  list of number of visits
             ucb_table: (`np.array`): [tot_visits, act_visits] list

        """

        assert (
            action_values.shape == action_visits.shape
        ), "expect the same number of values and # visits"
        assert np.all(
            action_visits >= 0
        ), f"visits must be positive, are not: {action_visits}"

        total_visits = action_visits.sum()
        assert total_visits > 0, "must have at least visited once for ucb"

        return action_values + ucb_table[total_visits, action_visits]
