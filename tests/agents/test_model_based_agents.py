"""tests `agents/model_based_agents.py` module"""
import unittest

from po_nrl.agents.model_based_agents import create_rollout_policy
from po_nrl.domains.gridverse_domain import GridverseDomain


class TestCreateRolloutPolicy(unittest.TestCase):
    """tests the factory function for rollouts"""

    def test_return_null(self):
        """tests default return: none"""
        self.assertIsNone(create_rollout_policy(None, ""))  # type: ignore

        d = GridverseDomain()
        self.assertIsNone(create_rollout_policy(d, ""))

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
