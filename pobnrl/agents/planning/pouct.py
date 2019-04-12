""" Particle Observable UCT """

from typing import Any
import copy
import random

import numpy as np

from agents.planning import ParticleFilter
from environments.environment import Environment


class TreeNode():
    """ node in the tree """

    def __init__(self, num_children: int, depth: int):
        """ creates a node with `num_children` childs at some depth `depth`

        Args:
             num_children: (`int`): the number of child nodes to have
             depth: (`int`): the depth of `this` in the tree

        """

        self.depth = depth

        self._children = [{} for _ in range(num_children)]

        # statistics
        self.avg_values = np.zeros(num_children)
        self.children_visits = np.zeros(num_children).astype(int)

    @property
    def num_children(self) -> int:
        """ returns the number of child nodes

        Args:

        RETURNS (`int`):

        """

        # probably faster than len() over a list
        return self.avg_values.shape[0]

    def child(self, action: int, observation: int) -> 'TreeNode':
        """ returns the child node associated with action-observation pair

        Args:
             action: (`int`): the chosen action
             observation: (`int`): the perceived observation

        RETURNS (`pobnrl.agents.planning.pouct.TreeNode`):

        """

        if observation not in self._children[action]:
            self._children[action][observation] \
                = TreeNode(len(self._children), depth=self.depth + 1)

        return self._children[action][observation]

    def update_value(self, child: int, value: float):
        """ updates the value `value` associated with child `child`

        Args:
             child: (`int`): the child to update
             value: (`float`): a new value to be added to the average

        """

        self.children_visits[child] += 1
        self.avg_values[child] += \
            (value - self.avg_values[child]) / self.children_visits[child]


class POUCT():
    """ MCTS for POMDPs using UCB """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            simulator: Environment,
            num_sims: int = 500,
            exploration_constant: float = 1.,
            planning_horizon: int = 10,
            discount: float = .95):
        """ TODO """

        self.num_sims = num_sims
        self.planning_horizon = planning_horizon
        self.discount = discount

        tot_visits = action_visits = np.arange(self.num_sims).reshape(1, -1)

        """
        precompute ucb table. table is such that:
        [m,n] = ucb for m total visits and n action visits
        rows: [[tot_visit=0],[tot_visit=2]...]
        where colums = [act_visit=0, act_visit=1 ...]
        """

        # + 1 for UCB1 (otherwise after trying 1 action all bonus == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            self._ucb_table = exploration_constant \
                * np.sqrt(np.log(tot_visits + 1).T / action_visits)

        self.simulator = simulator

    def select_action(self, belief: ParticleFilter):
        """ select_action

        Args:
             belief: (`pobnrl.agents.planning.beliefs.ParticleFilter`): the belief at the root to plan from

        """

        root = TreeNode(self.simulator.action_space.n, depth=0)

        # build tree
        for _ in range(self.num_sims):
            self.simulator.state = copy.deepcopy(belief.sample())
            self._traverse_tree(root)

        # pick best action from root
        return np.argmax(root.avg_values)

    def _traverse_tree(self, node: TreeNode) -> float:
        """ TODO """

        if node.depth == self.planning_horizon:
            return 0

        if node.num_visits == 0:
            ret = self._rollout(self.planning_horizon - node.depth)
        else:

            # UCB
            ucbs = POUCT.ucb(
                node.avg_values, node.children_visits, self._ucb_table
            )
            action = random.choice(np.argwhere(ucbs == ucbs.max()).flatten())

            step = self.simulator.step(action)
            ret = step.reward + self.discount * self._traverse_tree(
                node.child(action, self.simulator.obs2index(step.observation))
            )

        node.update_value(action, ret)

        return ret

    def _rollout(self, hor: int) -> float:
        """ TODO """

        ret = 0
        discount = 1

        for _ in range(hor):

            action = self.simulator.action_space.sample()
            step = self.simulator.step(action)

            ret += discount * step.reward

            discount *= self.discount

            if step.terminal:
                break

        return ret

    @staticmethod
    def ucb(
            action_values: np.array,  # of type floats
            action_visits: np.array,  # of type int
            ucb_table: np.ndarray) -> np.array:  # of type floats
        """ TODO """

        assert action_values.shape == action_visits.shape,\
            "expect the same number of values and # visits"
        assert np.all(action_visits >= 0), \
            f"visits must be positive, are not: {action_visits}"

        total_visits = action_visits.sum()
        assert total_visits > 0, "must have at least visited once for ucb"

        return action_values + ucb_table[total_visits, action_visits]
