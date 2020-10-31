"""tests `agents/model_based_agents.py` module"""
import unittest

from po_nrl.agents.model_based_agents import create_rollout_policy
from po_nrl.domains import GridverseDomain, Tiger
from po_nrl.environments import EncodeType


class TestCreateRolloutPolicy(unittest.TestCase):
    """tests the factory function for rollouts"""

    def test_return_random(self):
        """tests default return: none"""
        # pylint: disable=no-member
        d = Tiger(EncodeType.DEFAULT)

        self.assertEqual(
            create_rollout_policy(d, "").func.__name__,  # type: ignore
            "random_policy",
        )

        self.assertRaises(ValueError, create_rollout_policy, d, "default")
        self.assertRaises(ValueError, create_rollout_policy, d, "bla")

    def test_gridverse_policies(self):
        """tests rollouts for gridverse domain"""
        # pylint: disable=no-member
        d = GridverseDomain()

        self.assertRaises(ValueError, create_rollout_policy, d, "bla")

        self.assertEqual(
            create_rollout_policy(d, "default").func.__name__,  # type: ignore
            "default_rollout_policy",
        )
        self.assertEqual(
            create_rollout_policy(d, "gridverse-extra").func.__name__,  # type: ignore
            "straight_or_turn_policy",
        )


if __name__ == '__main__':
    unittest.main()
