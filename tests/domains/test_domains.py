"""Tests :mod:`general_bayes_adaptive_pomdps.domains`"""

import random

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.core import TerminalState
from general_bayes_adaptive_pomdps.domains import (
    CollisionAvoidance,
    GridWorld,
    RoadRacer,
    Tiger,
    collision_avoidance,
    gridworld,
    road_racer,
    tiger,
)
from general_bayes_adaptive_pomdps.domains.collision_avoidance import (
    CollisionAvoidancePrior,
)
from general_bayes_adaptive_pomdps.domains.gridworld import GridWorldPrior
from general_bayes_adaptive_pomdps.domains.road_racer import RoadRacerPrior
from general_bayes_adaptive_pomdps.domains.tiger import TigerPrior
from general_bayes_adaptive_pomdps.misc import DiscreteSpace


class TestTiger:
    """tests functionality of :class:`Tiger`"""

    def setup_method(self):
        """ creates a tiger member """
        self.one_hot_env = tiger.Tiger(one_hot_encode_observation=True)
        self.one_hot_env.reset()

        self.env = tiger.Tiger(one_hot_encode_observation=False)
        self.env.reset()

    def test_reset(self):
        """ tests that start state is 0 or 1 """

        assert self.one_hot_env.state in [0, 1]

        states = [self.one_hot_env.sample_start_state() for _ in range(0, 20)]

        for state in states:
            assert state[0] in [0, 1], "state should be either 0 or 1"

        assert [0] in states, "there should be at least one of this state"
        assert [1] in states, "there should be at least one of this state"

        # one-hot observation
        obs = [self.one_hot_env.reset() for _ in range(0, 10)]
        for observation in obs:
            np.testing.assert_array_equal(observation, [1, 1])

        # regular observation
        obs = [self.env.reset() for _ in range(0, 10)]
        for observation in obs:
            np.testing.assert_array_equal(observation, [2])

    def test_step(self):
        """ tests some basic dynamics """

        state = self.one_hot_env.state

        obs = []
        # tests effect of listening
        for _ in range(0, 50):
            step = self.one_hot_env.step(self.one_hot_env.LISTEN)
            np.testing.assert_array_equal(state, self.one_hot_env.state)
            assert step.observation.tolist() in [[0, 1], [1, 0]]
            assert not step.terminal
            assert step.reward == -1.0

            obs.append(step.observation.tolist())

        # tests stochasticity of observations when listening
        assert [0, 0] not in obs
        assert [1, 1] not in obs
        assert [0, 1] in obs
        assert [1, 0] in obs

        # test opening correct door
        for _ in range(0, 5):
            self.one_hot_env.reset()
            open_correct_door = self.one_hot_env.state[0]  # implementation knowledge
            step = self.one_hot_env.step(open_correct_door)

            np.testing.assert_array_equal(step.observation, [1, 1])
            assert step.reward == 10
            assert step.terminal

        # test opening correct door
        for _ in range(0, 5):
            self.one_hot_env.reset()
            open_wrong_door = 1 - self.one_hot_env.state[0]  # implementation knowledge
            step = self.one_hot_env.step(open_wrong_door)

            np.testing.assert_array_equal(step.observation, [1, 1])
            assert step.reward == -100
            assert step.terminal

    def test_sample_start_state(self):
        """ tests sampling start states """

        start_states = [self.one_hot_env.sample_start_state() for _ in range(10)]

        assert [0] in start_states
        assert [1] in start_states

        for state in start_states:
            assert state in [[0], [1]]

    def test_space(self):
        """ tests the size of the spaces """

        action_space = self.one_hot_env.action_space
        np.testing.assert_array_equal(action_space.size, [3])
        assert action_space.n == 3

        one_hot_observation_space = self.one_hot_env.observation_space
        np.testing.assert_array_equal(one_hot_observation_space.size, [2, 2])
        assert one_hot_observation_space.n == 4

        observation_space = self.env.observation_space
        np.testing.assert_array_equal(observation_space.size, [3])
        assert observation_space.n == 3

    def test_observation_projection(self):
        """ tests tiger.obs2index """

        assert self.one_hot_env.obs2index(self.one_hot_env.reset()) == 2
        assert self.one_hot_env.obs2index(self.one_hot_env.reset()) == 2

        assert self.one_hot_env.obs2index(np.array([1, 0])) == 0
        assert self.one_hot_env.obs2index(np.array([0, 1])) == 1

        assert self.one_hot_env.obs2index(np.array([0, 0])) == -1
        assert self.one_hot_env.obs2index(np.array([1, 1])) == 2

    def test_observation_encoding(self):
        """ tests encoding of observation in Tiger """

        np.testing.assert_array_equal(self.one_hot_env.encode_observation(0), [1, 0])
        np.testing.assert_array_equal(self.one_hot_env.encode_observation(1), [0, 1])
        np.testing.assert_array_equal(self.one_hot_env.encode_observation(2), [1, 1])

        np.testing.assert_array_equal(self.env.encode_observation(0), [0])
        np.testing.assert_array_equal(self.env.encode_observation(1), [1])
        np.testing.assert_array_equal(self.env.encode_observation(2), [2])

    def test_tiger_obs_prob(self):
        """ tests changing the observation probability """

        deterministic_tiger = tiger.Tiger(
            one_hot_encode_observation=False, correct_obs_probs=[1.0, 1.0]
        )

        deterministic_tiger.reset()

        step_result = deterministic_tiger.step(action=deterministic_tiger.LISTEN)

        assert step_result.observation[0] == deterministic_tiger.state[0]


class TestGridWorld:
    """Tests for GridWorld"""

    def test_reset(self):
        """ Tests Gridworld.reset() """
        env = gridworld.GridWorld(3, one_hot_encode_goal=True)
        observation = env.reset()

        np.testing.assert_array_equal(env.state[0], [0, 0])
        assert observation[2:].astype(int).tolist() in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

        env = gridworld.GridWorld(3, one_hot_encode_goal=False)
        observation = env.reset()
        assert observation[2] in [0, 1, 2]

    def test_sample_start_state(self):
        """ tests sampling start states """

        env = gridworld.GridWorld(5, one_hot_encode_goal=True)

        start_states = [env.sample_start_state() for _ in range(10)]

        for state in start_states:
            np.testing.assert_array_equal(state[0], [0, 0])

    def test_step(self):
        """ Tests Gridworld.step() """

        env = gridworld.GridWorld(3, one_hot_encode_goal=True)
        env.reset()

        goal = env.state[2]

        # left and low keeps in place
        actions = [env.WEST, env.SOUTH]
        for _ in range(10):
            step = env.step(np.random.choice(actions))

            np.testing.assert_array_equal(env.state[:2], [0, 0])
            assert step.reward == 0
            assert not step.terminal
            assert env.state[2] == goal

        # test step up
        step = env.step(env.NORTH)

        assert isinstance(step.observation[0], np.int64)
        assert env.state[:2].tolist() in [[0, 0], [0, 1]]
        assert not step.terminal
        assert step.reward == 0
        assert env.state[2] == goal

        # test step east
        env.reset()
        goal = env.state[2]
        step = env.step(env.EAST)

        assert env.state[:2].tolist() in [[0, 0], [1, 0]]
        assert not step.terminal
        assert step.reward == 0
        assert env.state[2] == goal

        random_goal = random.choice(env.goals)
        env.state = np.array([random_goal.x, random_goal.y, random_goal.index])
        step = env.step(np.random.randint(0, 4))

        assert step.reward == 1
        assert step.terminal

    def test_space(self):
        """ Tests Gridworld.spaces """
        env = gridworld.GridWorld(5, one_hot_encode_goal=True)

        action_space = env.action_space
        np.testing.assert_array_equal(action_space.size, [4])

        # one-hot goal encoding
        observation_space = env.observation_space
        assert observation_space.n == 25 * pow(2, len(env.goals))
        assert observation_space.ndim == 2 + len(env.goals)

        # regular (not one-hot) goal encoding
        env = gridworld.GridWorld(5, one_hot_encode_goal=False)
        observation_space = env.observation_space
        assert observation_space.n == 25 * len(env.goals)
        assert observation_space.ndim == 3

    def test_utils(self):
        """Tests misc functionality in Gridworld

        * Gridworld.generate_observation()
        * Gridworld.bound_in_grid()
        * Gridworld.sample_goal()
        * Gridworld.goals
        * Gridworld.space functionality
        * Gridworld.size

        """

        env = gridworld.GridWorld(3, one_hot_encode_goal=True)

        # test bound_in_grid
        np.testing.assert_array_equal(env.bound_in_grid(np.array([2, 2])), [2, 2])

        np.testing.assert_array_equal(env.bound_in_grid(np.array([3, 2])), [2, 2])

        np.testing.assert_array_equal(env.bound_in_grid(np.array([-1, 3])), [0, 2])

        list_of_goals = [
            gridworld.GridWorld.Goal(1, 2, 0),
            gridworld.GridWorld.Goal(2, 1, 1),
            gridworld.GridWorld.Goal(2, 2, 2),
        ]

        # test sample_goal
        assert env.sample_goal() in list_of_goals

        # Gridworld.goals
        assert env.goals == list_of_goals

        larger_env = gridworld.GridWorld(7, one_hot_encode_goal=True)
        assert len(larger_env.goals) == 10

        # Gridworld.state
        random_goal = random.choice(env.goals)
        test_state = np.array([random_goal.x, random_goal.y, random_goal.index])
        env.state = test_state
        np.testing.assert_array_equal(env.state, test_state)

        with pytest.raises(AssertionError):
            env.state = np.array([3, 2, random_goal.index])

        with pytest.raises(AssertionError):
            env.state = np.array([0, 2, 5])

        # Gridworld.size
        assert env.size == 3
        assert larger_env.size == 7

        # generate_observation
        np.testing.assert_array_almost_equal(
            env.obs_mult, [0.05, 0.05, 0.8, 0.05, 0.05]
        )
        env.obs_mult = np.array([0, 0, 1, 0, 0])

        observation = env.generate_observation(np.array([0, 0, 2]))
        np.testing.assert_array_equal(observation[2:], [0, 0, 1])
        np.testing.assert_array_equal(observation[:2], [0, 0])

        observation = env.generate_observation(np.array([1, 1, 1]))
        np.testing.assert_array_equal(observation[2:], [0, 1, 0])
        np.testing.assert_array_equal(observation[:2], [1, 1])

        env.obs_mult = np.array([0, 0.5, 0, 0.5, 0])
        observation = env.generate_observation(np.array([0, 0, 2]))
        assert observation[0] in [0, 1]
        assert observation[1] in [0, 1]

        observation = env.generate_observation(np.array([1, 1, 2]))
        assert observation[0] in [0, 2]
        assert observation[1] in [0, 2]

        env.obs_mult = np.array([1, 0, 0, 0, 0])
        observation = env.generate_observation(np.array([1, 2, 2]))
        assert observation[0] == 0
        assert observation[1] == 0

    def test_given_slow_cells(self) -> None:
        """ tests creation of gridworld without slow cells """

        gw_no_cells = gridworld.GridWorld(
            domain_size=5, one_hot_encode_goal=False, slow_cells=set()
        )
        assert not gw_no_cells.slow_cells

        gw_simple_cell = gridworld.GridWorld(
            domain_size=5, one_hot_encode_goal=False, slow_cells={(0, 1)}
        )
        assert gw_simple_cell.slow_cells == {(0, 1)}

    def test_goals(self) -> None:
        """ tests the location of goals """

        grid_world = gridworld.GridWorld(3, one_hot_encode_goal=True)

        for goal in grid_world.goals:
            assert goal.x >= 0
            assert goal.y >= 0
            assert goal.index >= 0


class TestCollisionAvoidance:
    """ Tests Collision Avoidance class """

    def test_reset(self):
        """ tests CollisionAvoidance.reset """

        env = collision_avoidance.CollisionAvoidance(3)
        env.reset()

        assert env.state[0] == 2
        assert env.state[1] == 1
        assert env.state[2] == 1

    def test_sample_start_state(self):
        """ tests sampling start states """

        env = collision_avoidance.CollisionAvoidance(7)
        np.testing.assert_array_equal(env.sample_start_state(), [6, 3, 3])

    def test_step(self):
        """ tests CollisionAvoidance.step """

        env = collision_avoidance.CollisionAvoidance(7)

        env.reset()
        step = env.step(1)
        assert isinstance(step.observation[0], np.int64)
        assert env.state[0] == 5
        assert env.state[1] == 3
        assert not step.terminal
        assert step.reward == 0

        env.reset()
        step = env.step(0)
        assert env.state[0] == 5
        assert env.state[1] == 2
        assert not step.terminal
        assert step.reward == -1

        env.reset()
        step = env.step(2)
        assert env.state[0] == 5
        assert env.state[1] == 4
        assert not step.terminal
        assert step.reward == -1

        step = env.step(2)
        assert env.state[0] == 4
        assert env.state[1] == 5
        assert not step.terminal
        assert step.reward == -1

        env.step(1)
        env.step(1)
        env.step(1)
        step = env.step(1)

        assert env.state[0] == 0
        assert env.state[1] == 5
        assert step.terminal

        # make sure terminal state raises the corrct error
        with pytest.raises(TerminalState):
            env.step(0)

        should_be_rew = -1000 if env.state[2] == 5 else 0
        assert step.reward == should_be_rew

        up_env = collision_avoidance.CollisionAvoidance(7, (0, 0, 1))
        up_env.step(1)
        assert up_env.state[2] == 4
        up_env.step(1)
        assert up_env.state[2] == 5
        up_env.step(1)
        up_env.step(1)
        assert up_env.state[2] == 6

        down_env = collision_avoidance.CollisionAvoidance(7, (1, 0, 0))
        down_env.step(1)
        assert down_env.state[2] == 2
        down_env.step(1)
        assert down_env.state[2] == 1
        down_env.step(1)
        down_env.step(1)
        assert down_env.state[2] == 0

    def test_utils(self):
        """Tests misc functionality in Collision Avoidance

        * CollisionAvoidance.generate_observation()
        * CollisionAvoidance.bound_in_grid()
        * CollisionAvoidance.action_sace
        * CollisionAvoidance.observation_space
        * CollisionAvoidance.size

        """

        env = collision_avoidance.CollisionAvoidance(7)

        # action_space
        assert env.action_space.n == 3
        assert env.action_space.ndim == 1
        np.testing.assert_array_equal(env.action_space.size, [3])

        # observation_space
        assert env.observation_space.n == 7 * 7 * 7
        assert env.observation_space.ndim == 3
        np.testing.assert_array_equal(env.observation_space.size, [7, 7, 7])

        # bound_in_grid
        assert env.bound_in_grid(8) == 6
        assert env.bound_in_grid(2) == 2
        assert env.bound_in_grid(-2) == 0

        # size
        assert env.size == 7

        obs = env.generate_observation()
        assert obs.shape == (3,)
        np.testing.assert_array_equal(obs[:2], [6, 3])
        assert obs[2] in list(range(7))

        obs = env.generate_observation(np.array([2, 4, 3]))
        assert obs.shape == (3,)
        np.testing.assert_array_equal(obs[:2], [2, 4])
        assert obs[2] in list(range(7))


class TestRoadRacer:
    """ tests the road race domain """

    def setup_method(self):
        """ creates a random domain """

        self.length = 6
        self.num_lanes = int(np.random.choice(range(3, 15, 2)))
        self.probs = np.random.rand(self.num_lanes)

        self.random_env = road_racer.RoadRacer(self.probs)
        self.random_env.reset()

    def test_init(self) -> None:
        """some tests on the `general_bayes_adaptive_pomdps.domains.road_racer.RoadRacer` init

        raises on   * even lanes
                    * probs outside of 0 and 1
                    * less than 2 long

        """

        with pytest.raises(AssertionError):
            road_racer.RoadRacer(np.array([0.5, 0.7]))
        with pytest.raises(AssertionError):
            road_racer.RoadRacer(np.array([0.5, 0.7, 1.1]))
        with pytest.raises(AssertionError):
            road_racer.RoadRacer(np.array([0.5, 0.7, -0.1]))

    def test_properties(self) -> None:
        """tests basic properties


        `general_bayes_adaptive_pomdps.domains.road_racer.RoadRacer.num_lanes`
        `general_bayes_adaptive_pomdps.domains.road_racer.RoadRacer.current_lane`
        `general_bayes_adaptive_pomdps.domains.road_racer.RoadRacer.middle_lane`
        `general_bayes_adaptive_pomdps.domains.road_racer.RoadRacer.action_space`
        `general_bayes_adaptive_pomdps.domains.road_racer.RoadRacer.observation_space`
        `general_bayes_adaptive_pomdps.domains.road_racer.RoadRacer.state_space`

        """

        # random domain properties
        assert self.random_env.lane_length == self.length
        assert self.random_env.num_lanes == self.num_lanes
        assert self.random_env.current_lane == (self.num_lanes - 1) / 2
        assert self.random_env.middle_lane == (self.num_lanes - 1) / 2

        assert self.random_env.action_space.n == 3

        obs_space = self.random_env.observation_space
        assert isinstance(obs_space, DiscreteSpace)
        np.testing.assert_array_equal(obs_space.size, [self.length])

        state_space = self.random_env.state_space
        assert isinstance(state_space, DiscreteSpace)
        np.testing.assert_array_equal(
            state_space.size,
            [self.length] * (self.num_lanes) + [self.num_lanes],
        )

    def test_functional(self) -> None:
        """tests the pure functions defined in the road racer package

        `general_bayes_adaptive_pomdps.domains.road_racer.current_lane`
        `general_bayes_adaptive_pomdps.domains.road_racer.get_observation`


        """

        state = np.random.randint(
            low=2,
            high=np.random.randint(low=3, high=5),
            size=np.random.randint(low=3, high=5),
        )

        state[-1] = 0
        assert road_racer.RoadRacer.get_current_lane(state) == 0

        state[-1] = 3
        assert road_racer.RoadRacer.get_current_lane(state) == 3

        state[-1] = -1
        with pytest.raises(AssertionError):
            road_racer.RoadRacer.get_current_lane(state)

        dist = state[0]
        state[-1] = 0
        np.testing.assert_array_equal(self.random_env.get_observation(state), [dist])

        dist = state[1]
        state[-1] = 1
        np.testing.assert_array_equal(
            self.random_env.get_observation(state), [dist], f"state={state}"
        )

    def test_reset(self) -> None:
        """tests resetting the domain

        Args:

        RETURNS (`None`):

        """
        np.testing.assert_array_equal(self.random_env.reset(), [self.length - 1])

    def test_start_start(self) -> None:
        """tests whether the start state are sampled correctly

        Args:

        RETURNS (`None`):

        """

        start_state = np.ones(self.num_lanes + 1) * self.length - 1
        start_state[-1] = int(self.num_lanes / 2)

        np.testing.assert_array_equal(self.random_env.sample_start_state(), start_state)

    def test_reward(self) -> None:
        """test whether rewards are generated correctly

        Args:

        RETURNS (`None`):

        """

        state = np.random.randint(low=2, high=self.length - 1, size=self.num_lanes + 1)

        # put agent in first lane
        state[-1] = 0

        next_state = self.random_env.simulation_step(
            state, road_racer.RoadRacer.NO_OP
        ).state

        dist = next_state[0]

        assert (
            self.random_env.reward(
                state, road_racer.RoadRacer.NO_OP, new_state=next_state
            )
            == dist
        ), f"{state} -> {next_state}"

        # imagine agent tried to go up -> penalty of 1
        assert (
            self.random_env.reward(
                state, road_racer.RoadRacer.GO_UP, new_state=next_state
            )
            == dist - 1
        ), f"{state} -> {next_state}"

        # imagine agent went down
        next_state[-1] = 1
        dist = next_state[1]
        assert (
            self.random_env.reward(
                state, road_racer.RoadRacer.GO_DOWN, new_state=next_state
            )
            == dist
        ), f"{state} -> {next_state}"

        # imagine agent **tried** to go down but was not possible
        next_state[1] = 0
        next_state[-1] = 0
        dist = next_state[0]
        assert (
            self.random_env.reward(
                state, road_racer.RoadRacer.GO_DOWN, new_state=next_state
            )
            == dist - 1
        ), f"{state} -> {next_state}"

    def test_terminal(self) -> None:
        """tests that it never returns terminal

        Args:

        RETURNS (`None`):

        """

        # test any random <s,a,s'> transition
        assert not self.random_env.terminal(
            np.concatenate(
                (
                    np.random.randint(low=1, high=self.length, size=self.num_lanes),
                    [random.randint(0, self.num_lanes - 1)],
                )
            ),
            self.random_env.action_space.sample_as_int(),
            np.concatenate(
                (
                    np.random.randint(low=1, high=self.length, size=self.num_lanes),
                    [random.randint(0, self.num_lanes - 1)],
                )
            ),
        )

    def test_steps(self) -> None:
        """tests the actual stepping functionality

        Args:

        RETURNS (`None`):

        """

        go_up = road_racer.RoadRacer.GO_UP
        stay = road_racer.RoadRacer.NO_OP
        go_down = road_racer.RoadRacer.GO_DOWN

        # test regular step: all lanes should either stay or advance one
        state = np.random.randint(low=2, high=self.length - 1, size=self.num_lanes + 1)
        state[-1] = 1

        next_state = self.random_env.simulation_step(state, stay).state
        for feature, value in enumerate(next_state):
            assert value in [
                state[feature],
                state[feature] - 1,
            ], f"{state} -> {next_state}, failing at feature {feature}"

        # test that lane advance p=1 will advance, and p=0 will stay
        self.random_env.lane_probs = np.ones(self.num_lanes)

        next_state = state - 1
        next_state[-1] = state[-1]

        np.testing.assert_array_equal(
            self.random_env.simulation_step(state, stay).state, next_state
        )

        self.random_env.lane_probs = np.zeros(self.num_lanes)
        np.testing.assert_array_equal(
            self.random_env.simulation_step(state, stay).state, state
        )

        # test changing lane
        state[-1] = 0
        assert self.random_env.simulation_step(state, go_down).state[-1] == 1

        state[-1] = self.num_lanes - 1
        assert (
            self.random_env.simulation_step(state, go_up).state[-1]
            == self.num_lanes - 2
        )

        state[-1] = 1
        assert self.random_env.simulation_step(state, go_down).state[-1] == 2
        assert self.random_env.simulation_step(state, go_up).state[-1] == 0

        # test changing lane on border will not change
        state[-1] = 0
        assert self.random_env.simulation_step(state, go_up).state[-1] == 0

        state[-1] = self.num_lanes - 1
        assert (
            self.random_env.simulation_step(state, go_down).state[-1]
            == self.num_lanes - 1
        )

        # test that changing lane into a car will not change lane
        state[-1] = 0
        assert self.random_env.simulation_step(state, go_down).state[-1] == 1

        state[-1] = self.num_lanes - 1
        assert (
            self.random_env.simulation_step(state, go_up).state[-1]
            == self.num_lanes - 2
        )

        state[-1] = 1
        assert self.random_env.simulation_step(state, go_down).state[-1] == 2
        assert self.random_env.simulation_step(state, go_up).state[-1] == 0

        # test cars starting in lane again after being passed
        self.random_env.lane_probs = np.ones(self.num_lanes)
        state[0] = 0
        state[-1] = 1
        np.testing.assert_array_equal(
            self.random_env.simulation_step(state, stay).state[0],
            self.length - 1,
        )

        # test lane will not advance if agent is blocking
        state[1] = 1
        np.testing.assert_array_equal(
            self.random_env.simulation_step(state, stay).state[1], 1
        )

        # test throwing terminal state if given state where agent is on car
        state[road_racer.RoadRacer.get_current_lane(state)] = 0
        with pytest.raises(TerminalState):
            self.random_env.simulation_step(state, stay)


class TestRoadRacerPrior:
    """ tests the road race prior """

    def test_lane_probs(self) -> None:
        """ some basic tests """

        # test .5 probabilities with certainty
        domain = RoadRacerPrior(3, 1000000000).sample()
        assert isinstance(domain, RoadRacer)

        np.testing.assert_almost_equal(
            domain.lane_probs, np.array([0.5, 0.5, 0.5]), decimal=3
        )

        # test always within 0 and 1
        domain = RoadRacerPrior(5, 0.5).sample()
        assert isinstance(domain, RoadRacer)

        assert np.all(domain.lane_probs > 0), f"probs={domain.lane_probs}"
        assert np.all(domain.lane_probs < 1), f"probs={domain.lane_probs}"


class TestCollisionAvoidancePrior:
    """Tests the prior on collection avoidance"""

    def test_default(self) -> None:
        """ tests the default prior """

        sampled_domain = CollisionAvoidancePrior(3, 1).sample()

        assert isinstance(sampled_domain, CollisionAvoidance)

        block_pol = sampled_domain._block_policy

        np.testing.assert_array_less(block_pol, 1)
        np.testing.assert_array_less(0, block_pol)

        assert round(abs(np.sum(block_pol) - 1), 7) == 0

    def test_certain_prior(self) -> None:
        """ very certain prior """
        sampled_domain = CollisionAvoidancePrior(3, 10000000).sample()

        assert isinstance(sampled_domain, CollisionAvoidance)

        block_pol = sampled_domain._block_policy

        assert round(abs(block_pol[0] - 0.05), 3) == 0
        assert round(abs(block_pol[1] - 0.9), 3) == 0
        assert round(abs(block_pol[2] - 0.05), 3) == 0

    def test_uncertain_prior(self) -> None:
        """ uncertain prior """
        sampled_domain = CollisionAvoidancePrior(3, 10).sample()

        assert isinstance(sampled_domain, CollisionAvoidance)

        block_pol = sampled_domain._block_policy

        assert round(abs(block_pol[0] - 0.05), 6) != 0
        assert round(abs(block_pol[1] - 0.9), 6) != 0
        assert round(abs(block_pol[2] - 0.05), 6) != 0


class TestGridWorldPrior:
    """ tests the prior over the gridworld problem """

    def test_encoding(self) -> None:
        """ tests encoding is done correctly """

        one_hot_sample = GridWorldPrior(size=3, one_hot_encode_goal=True).sample()
        assert one_hot_sample.observation_space.ndim == 5

        default_sample = GridWorldPrior(size=5, one_hot_encode_goal=False).sample()
        assert default_sample.observation_space.ndim == 3

    def test_default_slow_cells(self) -> None:
        """ tests Gridworlds sampled from prior have no slow cells """

        sample_gridworld_1 = GridWorldPrior(size=4, one_hot_encode_goal=False).sample()
        sample_gridworld_2 = GridWorldPrior(size=4, one_hot_encode_goal=False).sample()

        assert isinstance(sample_gridworld_1, GridWorld)
        assert isinstance(sample_gridworld_2, GridWorld)

        assert sample_gridworld_1.slow_cells, "may **rarely** be empty"
        assert sample_gridworld_2.slow_cells, "may **rarely** be empty"
        assert (
            sample_gridworld_1.slow_cells != sample_gridworld_2.slow_cells
        ), "may **rarely** be true"


class TestTigerPrior:
    """ tests that the observation probability is reasonable """

    def test_encoding(self) -> None:
        """ tests the sample method encoding is correct """

        num_total_counts = 10.0
        incorrect_prior_setting = 0.0

        one_hot_prior = TigerPrior(
            num_total_counts, incorrect_prior_setting, one_hot_encode_observation=True
        )
        assert one_hot_prior.sample().observation_space.ndim == 2

        default_prior = TigerPrior(
            num_total_counts, incorrect_prior_setting, one_hot_encode_observation=False
        )
        assert default_prior.sample().observation_space.ndim == 1

    def test_observation_prob(self) -> None:
        """ tests the observation probability of samples """

        num_total_counts = 10.0
        incorrect_prior_setting = 0.0

        encoding = random.choice([True, False])

        tiger = TigerPrior(
            num_total_counts,
            incorrect_prior_setting,
            one_hot_encode_observation=encoding,
        ).sample()
        assert isinstance(tiger, Tiger)

        obs_probs = tiger._correct_obs_probs
        assert 0 <= obs_probs[0] <= 1
        assert 0 <= obs_probs[1] <= 1

    def test_prior_correctness(self) -> None:
        """tests the prior correctness parameter

        Values tested: .1, .5, .9999, 1.

        Args:

        RETURNS (`None`):

        """
        num_total_counts = 1000000.0
        prior_level = 0.9999

        encoding = random.choice([True, False])

        tiger = TigerPrior(num_total_counts, prior_level, encoding).sample()
        assert isinstance(tiger, Tiger)

        obs_probs = tiger._correct_obs_probs
        assert 0.84 <= obs_probs[0] <= 0.86
        assert 0.84 <= obs_probs[1] <= 0.86

        prior_level = 0.5
        tiger = TigerPrior(num_total_counts, prior_level, encoding).sample()

        assert isinstance(tiger, Tiger)
        obs_probs = tiger._correct_obs_probs
        assert 0.73 <= obs_probs[0] <= 0.74
        assert 0.73 <= obs_probs[1] <= 0.74

        prior_level = 0.1
        tiger = TigerPrior(num_total_counts, prior_level, encoding).sample()
        assert isinstance(tiger, Tiger)
        obs_probs = tiger._correct_obs_probs
        assert 0.64 <= obs_probs[0] <= 0.65
        assert 0.64 <= obs_probs[1] <= 0.65

        prior_level = 1.01
        with pytest.raises(ValueError):
            TigerPrior(num_total_counts, prior_level, encoding)

        prior_level = -0.1
        with pytest.raises(ValueError):
            TigerPrior(num_total_counts, prior_level, encoding)

    def test_num_total_counts(self) -> None:
        """tests the parameter # of total counts

        Args:

        RETURNS (`None`):

        """

        high_total_counts = 10000000.0
        low_total_counts = 1.0
        incorrect_prior_setting = 0.0

        encoding = random.choice([True, False])

        tiger = TigerPrior(
            high_total_counts, incorrect_prior_setting, encoding
        ).sample()
        assert isinstance(tiger, Tiger)

        obs_probs = tiger._correct_obs_probs
        assert 0.624 <= obs_probs[0] <= 0.626
        assert 0.624 <= obs_probs[1] <= 0.626

        tiger = TigerPrior(low_total_counts, incorrect_prior_setting, encoding).sample()
        assert isinstance(tiger, Tiger)

        obs_probs = tiger._correct_obs_probs
        assert not (
            0.624 <= obs_probs[0] <= 0.626
        ), f"Rarely false; obs = {obs_probs[0]}"
        assert not (
            0.624 <= obs_probs[1] <= 0.626
        ), f"Rarely false; obs = {obs_probs[0]}"


if __name__ == "__main__":
    pytest.main([__file__])
