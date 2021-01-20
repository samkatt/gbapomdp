""" Particle Observable UCT """
from typing import Tuple

import numpy as np
import online_pomdp_planning.mcts as online_pomdp_planning_types
from online_pomdp_planning.mcts import Policy as Lib_Policy
from online_pomdp_planning.mcts import create_POUCT as lib_create_POUCT
from general_bayes_adaptive_pomdps.agents.planning.particle_filters import Belief
from general_bayes_adaptive_pomdps.environments import ActionSpace, Simulator
from typing_extensions import Protocol  # pylint: disable=wrong-import-order


class Planner(Protocol):
    """The signature of a planner in this package"""

    def __call__(self, belief: Belief) -> np.ndarray:
        """A planner is a function that picks an action given a belief

        Args:
            belief (Belief): the current belief (PF)

        Returns:
            int: the chosen action
        """


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


class OnlinePlanningSim(online_pomdp_planning_types.Simulator):
    """A simulator for online planner created from our domains"""

    def __init__(self, bnrl_simulator: Simulator):
        """Wraps and calls ``bnrl_simulator`` with imposed signature"""
        super().__init__()
        self._bnrl_sim = bnrl_simulator

        # XXX: ugly, throw-away code
        if hasattr(self._bnrl_sim, "simulation_step_without_updating_theta"):
            self.sim_step = self._bnrl_sim.simulation_step_without_updating_theta  # type: ignore
        else:
            self.sim_step = self._bnrl_sim.simulation_step

    def __call__(
        self, s: np.ndarray, a: int
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """The signature for the simulator for online planning

        Upon calling, produces a transition (state, observation, reward, terminal)

        Args:
            s (np.ndarray): input state
            a (int): input action

        Returns:
            Tuple[np.ndarray, np.ndarray, float, bool]: next simulated artifacts
        """
        next_s, obs = self.sim_step(s, a)
        reward = self._bnrl_sim.reward(s, a, next_s)
        terminal = self._bnrl_sim.terminal(s, a, next_s)

        return next_s, obs.data.tobytes(), reward, terminal


class OnlinePlanningPolicy(Lib_Policy):
    """A (rollout) policy for online planning created from our domains"""

    def __init__(self, pol: RolloutPolicy):
        """Takes in a policy from this package and turns into one for online planning"""
        super().__init__()
        self._rollout_pol = pol

    def __call__(self, s: np.ndarray, o: np.ndarray) -> int:
        """The signature for the policy for online planning

        A stochastic mapping from state and observation to action

        Args:
            s (np.ndarray): the state
            o (np.ndarray): the observation

        Returns:
            int: chosen action
        """
        return self._rollout_pol(s)


def create_planner(
    simulator: Simulator,
    rollout_policy: RolloutPolicy,
    num_sims: int = 500,
    exploration_constant: float = 1.0,
    planning_horizon: int = 10,
    discount: float = 0.95,
) -> Planner:
    """The factory function for planners

    Currently just returns PO-UCT with given parameters, but allows for future generalization

    Args:
        simulator (`general_bayes_adaptive_pomdps.environments.Simulator`): the simulator to plan against
        rollout_policy (`RolloutPolicy`): the rollout policy
        num_sims (`int`): number of simulations to run
        exploration_constant (`float`): the UCB-constant for UCB
        planning_horizon (`int`): how far into the future to plan for
        discount (`float`):

    Returns:
        Planner: A planner created given the configuration
    """

    actions = list(np.int64(i) for i in range(simulator.action_space.n))
    online_planning_sim = OnlinePlanningSim(simulator)
    policy = OnlinePlanningPolicy(rollout_policy)

    online_planner = lib_create_POUCT(
        actions,
        online_planning_sim,
        num_sims,
        policy=policy,
        discount_factor=discount,
        rollout_depth=planning_horizon,
        ucb_constant=exploration_constant,
    )

    return lambda b: online_planner(b)[0]
