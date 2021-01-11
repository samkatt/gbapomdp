""" runs tests on planners (PO-UCT)"""

import unittest

from po_nrl.agents.planning.pouct import random_policy
from po_nrl.environments import ActionSpace


class TestPOUCT(unittest.TestCase):
    """ class to test POUCT """

    def test_random_rollout(self):
        """tests `pouct.random_policy`"""

        space = ActionSpace(4)
        actions = {random_policy(_, space) for _ in range(20)}

        self.assertSetEqual(actions, {0, 1, 2, 3})


if __name__ == '__main__':
    unittest.main()
