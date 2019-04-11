""" Particle Observable UCT """

import copy
import random

import numpy as np

from agents.planning import ParticleFilter
from environments.environment import Environment


class POUCT():  # pylint: disable=too-few-public-methods
    """ MCTS for POMDPs using UCB """

    class Node():
        """ node in the tree """

        def __init__(self, num_children: int, depth: int):
            """ TODO """

            self.depth = depth

            self._children = [{} for _ in range(num_children)]

            # statistics
            self.avg_values = np.full(num_children, -float('inf'))
            self.children_visits = np.zeros(num_children)

        def child(self, action: int, observation: int):
            """

            TODO: doc & test

            """

            if observation not in self._children[action]:
                self._children[action][observation] \
                    = POUCT.Node(len(self._children), depth=self.depth + 1)

            return self._children[action][observation]

        def update_value(self, child: int, value: int):
            """ TODO """
            self.children_visits[child] += 1

            # update avg value

            raise NotImplementedError

    def __init__(  # pylint: disable=too-many-arguments
            self,
            num_sims: int,
            exploration_constant: float,
            planning_horizon: int,
            discount: float,
            simulator: Environment):
        """ TODO """

        self.num_sims = num_sims
        self.planning_horizon = planning_horizon
        self.discount = discount

        # create ucb table: [m,n] = ucb for m total visits and n action visits
        table = []

        # TODO: test
        visits = np.arange(self.num_sims)
        for total_visits in range(self.num_sims):
            table.append(
                exploration_constant
                * np.sqrt(np.log(total_visits + 1) / visits)
            )

        self._ucb_table = np.array(table)
        self._ucb_table[:, 0] = float('inf')  # no action visits -> max

        self.simulator = simulator
        raise NotImplementedError

    def select_action(self, belief: ParticleFilter):
        """ select_action

        Args:
             belief: (`pobnrl.agents.planning.beliefs.ParticleFilter`): the belief at the root to plan from

        """

        root = POUCT.Node(self.simulator.action_space.n, depth=0)

        # build tree
        for _ in range(self.num_sims):
            self.simulator.state = copy.deepcopy(belief.sample())
            self._traverse_tree(root)

        # pick best action from root
        return np.argmax(root.avg_values)

    def _traverse_tree(self, node: Node) -> float:
        """ TODO """

        if node.depth == self.planning_horizon:
            return 0

        if node.num_visits == 0:
            ret = self._rollout(self.planning_horizon - node.depth)
        else:

            # UCB
            ucbs = self._ucb(node.avg_values, node.children_visits)
            action = random.choice(np.argwhere(ucbs == ucbs.max()).flatten())

            step = self.simulator.step(action)
            ret = step.reward + self.discount * self._traverse_tree(
                node.child(action, step.observation)
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

    def _ucb(
            self,
            action_values: np.array,
            action_visits: np.array,
            ucb_table: np.ndarray = None) -> np.array:
        """ TODO """

        assert action_values.shape == action_visits.shape,\
            "expect the same number of values and # visits"

        if ucb_table is None:
            ucb_table = self._ucb_table

        total_visits = np.sum(action_visits)

        return np.array(
            [q + ucb_table[total_visits, n]
             for q, n in zip(action_values, action_visits)]
        )
