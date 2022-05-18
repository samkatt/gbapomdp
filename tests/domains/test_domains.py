"""Tests :mod:`general_bayes_adaptive_pomdps.domains`"""

import random

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.core import TerminalState
from general_bayes_adaptive_pomdps.domains.collision_avoidance import (
    CollisionAvoidance,
    CollisionAvoidancePrior,
)
from general_bayes_adaptive_pomdps.domains.gridworld import GridWorld, GridWorldPrior
from general_bayes_adaptive_pomdps.domains.road_racer import RoadRacer, RoadRacerPrior
from general_bayes_adaptive_pomdps.domains.tiger import (
    Tiger,
    TigerPrior,
    create_tabular_prior_counts,
)
from general_bayes_adaptive_pomdps.misc import DiscreteSpace


def setup_tiger(one_hot: bool) -> Tiger:
    """creates a tiger member"""
    domain = Tiger(one_hot_encode_observation=one_hot)
    domain.reset()
    return domain


def test_reset_tiger():
    """tests that start state is 0 or 1"""

    one_hot_env = setup_tiger(True)
    env = setup_tiger(False)

    assert one_hot_env.state in [0, 1]

    states = [one_hot_env.sample_start_state() for _ in range(0, 20)]

    for state in states:
        assert state[0] in [0, 1], "state should be either 0 or 1"

    assert [0] in states, "there should be at least one of this state"
    assert [1] in states, "there should be at least one of this state"

    # one-hot observation
    obs = [one_hot_env.reset() for _ in range(0, 10)]
    for observation in obs:
        np.testing.assert_array_equal(observation, [1, 1])

    # regular observation
    obs = [env.reset() for _ in range(0, 10)]
    for observation in obs:
        np.testing.assert_array_equal(observation, [2])


def test_step_tiger():
    """tests some basic dynamics"""

    one_hot_env = setup_tiger(True)

    state = one_hot_env.state

    obs = []
    # tests effect of listening
    for _ in range(0, 50):
        step = one_hot_env.step(one_hot_env.LISTEN)
        np.testing.assert_array_equal(state, one_hot_env.state)
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
        one_hot_env.reset()
        open_correct_door = one_hot_env.state[0]  # implementation knowledge
        step = one_hot_env.step(open_correct_door)

        np.testing.assert_array_equal(step.observation, [1, 1])
        assert step.reward == 10
        assert step.terminal

    # test opening correct door
    for _ in range(0, 5):
        one_hot_env.reset()
        open_wrong_door = 1 - one_hot_env.state[0]  # implementation knowledge
        step = one_hot_env.step(open_wrong_door)

        np.testing.assert_array_equal(step.observation, [1, 1])
        assert step.reward == -100
        assert step.terminal


def test_sample_start_state_tiger():
    """tests sampling start states"""

    one_hot_env = setup_tiger(True)

    start_states = [one_hot_env.sample_start_state() for _ in range(10)]

    assert [0] in start_states
    assert [1] in start_states

    for state in start_states:
        assert state in [[0], [1]]


def test_space_tiger():
    """tests the size of the spaces"""
    one_hot_env = setup_tiger(True)
    env = setup_tiger(False)

    action_space = one_hot_env.action_space
    np.testing.assert_array_equal(action_space.size, [3])
    assert action_space.n == 3

    one_hot_observation_space = one_hot_env.observation_space
    np.testing.assert_array_equal(one_hot_observation_space.size, [2, 2])
    assert one_hot_observation_space.n == 4

    observation_space = env.observation_space
    np.testing.assert_array_equal(observation_space.size, [3])
    assert observation_space.n == 3


def test_observation_projection_tiger():
    """tests tiger.obs2index"""

    one_hot_env = setup_tiger(True)

    assert one_hot_env.obs2index(one_hot_env.reset()) == 2
    assert one_hot_env.obs2index(one_hot_env.reset()) == 2

    assert one_hot_env.obs2index(np.array([1, 0])) == 0
    assert one_hot_env.obs2index(np.array([0, 1])) == 1

    assert one_hot_env.obs2index(np.array([0, 0])) == -1
    assert one_hot_env.obs2index(np.array([1, 1])) == 2


def test_observation_encoding_tiger():
    """tests encoding of observation in Tiger"""

    one_hot_env = setup_tiger(True)
    env = setup_tiger(False)

    np.testing.assert_array_equal(one_hot_env.encode_observation(0), [1, 0])
    np.testing.assert_array_equal(one_hot_env.encode_observation(1), [0, 1])
    np.testing.assert_array_equal(one_hot_env.encode_observation(2), [1, 1])

    np.testing.assert_array_equal(env.encode_observation(0), [0])
    np.testing.assert_array_equal(env.encode_observation(1), [1])
    np.testing.assert_array_equal(env.encode_observation(2), [2])


def test_tiger_obs_prob_tiger():
    """tests changing the observation probability"""

    deterministic_tiger = Tiger(
        one_hot_encode_observation=False, correct_obs_probs=[1.0, 1.0]
    )
    deterministic_tiger.reset()

    step_result = deterministic_tiger.step(action=deterministic_tiger.LISTEN)
    assert step_result.observation[0] == deterministic_tiger.state[0]


@pytest.mark.parametrize(
    "prior_correctness,prior_certainty",
    [
        (1, 10),
        (1, 25),
        (0, 100),
        (0.25, 100),
    ],
)
def test_tabular_prior_tiger(prior_correctness, prior_certainty):
    """Tests the creation of prior counts"""
    p = create_tabular_prior_counts(
        correctness=prior_correctness, certainty=prior_certainty
    )

    print(p.T[Tiger.LEFT, Tiger.LISTEN, Tiger.LEFT])

    # listening keeps tiger in place
    assert p.T[Tiger.LEFT, Tiger.LISTEN, Tiger.LEFT] > 0
    assert p.T[Tiger.RIGHT, Tiger.LISTEN, Tiger.RIGHT] > 0

    assert p.T[Tiger.LEFT, Tiger.LISTEN, Tiger.RIGHT] == 0
    assert p.T[Tiger.RIGHT, Tiger.LISTEN, Tiger.LEFT] == 0

    # opening leads to uniform distribution with high counts
    assert (
        p.T[Tiger.LEFT, Tiger.LEFT, Tiger.RIGHT]
        == p.T[Tiger.LEFT, Tiger.LEFT, Tiger.LEFT]
        == p.T[Tiger.LEFT, Tiger.RIGHT, Tiger.LEFT]
        == p.T[Tiger.LEFT, Tiger.RIGHT, Tiger.RIGHT]
        == p.T[Tiger.RIGHT, Tiger.LEFT, Tiger.RIGHT]
        == p.T[Tiger.RIGHT, Tiger.LEFT, Tiger.LEFT]
        == p.T[Tiger.RIGHT, Tiger.RIGHT, Tiger.LEFT]
        == p.T[Tiger.RIGHT, Tiger.RIGHT, Tiger.RIGHT]
    )

    assert p.T[Tiger.LEFT, Tiger.LEFT, Tiger.RIGHT] > 1000

    # opening leads to no-obs
    assert (
        p.O[Tiger.LEFT, Tiger.LEFT, Tiger.LEFT]
        == p.O[Tiger.LEFT, Tiger.RIGHT, Tiger.LEFT]
        == p.O[Tiger.RIGHT, Tiger.LEFT, Tiger.LEFT]
        == p.O[Tiger.RIGHT, Tiger.RIGHT, Tiger.LEFT]
        == p.O[Tiger.LEFT, Tiger.LEFT, Tiger.RIGHT]
        == p.O[Tiger.LEFT, Tiger.RIGHT, Tiger.RIGHT]
        == p.O[Tiger.RIGHT, Tiger.LEFT, Tiger.RIGHT]
        == p.O[Tiger.RIGHT, Tiger.RIGHT, Tiger.RIGHT]
        == 0
    )

    assert (
        p.O[Tiger.LEFT, Tiger.LEFT, Tiger.NO_OBS]
        == p.O[Tiger.LEFT, Tiger.RIGHT, Tiger.NO_OBS]
        == p.O[Tiger.RIGHT, Tiger.LEFT, Tiger.NO_OBS]
        == p.O[Tiger.RIGHT, Tiger.RIGHT, Tiger.NO_OBS]
        == p.O[Tiger.LEFT, Tiger.LEFT, Tiger.NO_OBS]
        == p.O[Tiger.LEFT, Tiger.RIGHT, Tiger.NO_OBS]
        == p.O[Tiger.RIGHT, Tiger.LEFT, Tiger.NO_OBS]
        == p.O[Tiger.RIGHT, Tiger.RIGHT, Tiger.NO_OBS]
        > 1
    )

    # listening has expected certainty
    expected_prob = 0.625 + (prior_correctness * 0.225)
    expected_correct_door_count = expected_prob * prior_certainty
    expected_wrong_door_count = (1 - expected_prob) * prior_certainty

    assert (
        p.O[Tiger.LISTEN, Tiger.LEFT, Tiger.LEFT]
        == p.O[Tiger.LISTEN, Tiger.RIGHT, Tiger.RIGHT]
        == expected_correct_door_count
    )
    assert (
        p.O[Tiger.LISTEN, Tiger.RIGHT, Tiger.LEFT]
        == p.O[Tiger.LISTEN, Tiger.RIGHT, Tiger.LEFT]
        == expected_wrong_door_count
    )


def test_reset_gridworld():
    """Tests :meth:`Gridworld.reset`"""
    env = GridWorld(3, one_hot_encode_goal=True)
    observation = env.reset()

    np.testing.assert_array_equal(env.state[0], [0, 0])
    assert observation[2:].astype(int).tolist() in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    env = GridWorld(3, one_hot_encode_goal=False)
    observation = env.reset()
    assert observation[2] in [0, 1, 2]


def test_sample_start_state_gridworld():
    """tests :meth:`GridWorld.sample_start_state`"""

    env = GridWorld(5, one_hot_encode_goal=True)

    start_states = [env.sample_start_state() for _ in range(10)]

    for state in start_states:
        np.testing.assert_array_equal(state[0], [0, 0])


def test_step_gridworld():
    """Tests :meth:`Gridworld.step`"""

    env = GridWorld(3, one_hot_encode_goal=True)
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


def test_space_gridworld():
    """Tests :meth:`Gridworld.spaces`"""
    env = GridWorld(5, one_hot_encode_goal=True)

    action_space = env.action_space
    np.testing.assert_array_equal(action_space.size, [4])

    # one-hot goal encoding
    observation_space = env.observation_space
    assert observation_space.n == 25 * pow(2, len(env.goals))
    assert observation_space.ndim == 2 + len(env.goals)

    # regular (not one-hot) goal encoding
    env = GridWorld(5, one_hot_encode_goal=False)
    observation_space = env.observation_space
    assert observation_space.n == 25 * len(env.goals)
    assert observation_space.ndim == 3


def test_utils_gridworld():
    """Tests misc functionality in Gridworld

    * :meth:`Gridworld.generate_observation`
    * :meth:`Gridworld.bound_in_grid`
    * :meth:`Gridworld.sample_goal`
    * :meth:`Gridworld.goals`
    * :meth:`Gridworld.space functionality`
    * :meth:`Gridworld.size`
    """

    env = GridWorld(3, one_hot_encode_goal=True)

    # test bound_in_grid
    assert env.bound_in_grid(2, 2) == (2, 2)
    assert env.bound_in_grid(3, 2) == (2, 2)
    assert env.bound_in_grid(-1, 3) == (0, 2)

    list_of_goals = [
        GridWorld.Goal(1, 2, 0),
        GridWorld.Goal(2, 1, 1),
        GridWorld.Goal(2, 2, 2),
    ]

    # test sample_goal
    assert env.sample_goal() in list_of_goals

    # Gridworld.goals
    assert env.goals == list_of_goals

    larger_env = GridWorld(7, one_hot_encode_goal=True)
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
    np.testing.assert_array_almost_equal(env.obs_mult, [0.05, 0.05, 0.8, 0.05, 0.05])
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


def test_given_slow_cells_gridworld():
    """tests creation of gridworld without slow cells"""

    gw_no_cells = GridWorld(domain_size=5, one_hot_encode_goal=False, slow_cells=set())
    assert not gw_no_cells.slow_cells

    gw_simple_cell = GridWorld(
        domain_size=5, one_hot_encode_goal=False, slow_cells={(0, 1)}
    )
    assert gw_simple_cell.slow_cells == {(0, 1)}


def test_goals_gridworld():
    """tests the location of goals"""

    grid_world = GridWorld(3, one_hot_encode_goal=True)

    for goal in grid_world.goals:
        assert goal.x >= 0
        assert goal.y >= 0
        assert goal.index >= 0


def test_reset_collision_avoidance():
    """tests :meth:`CollisionAvoidance.reset`"""

    env = CollisionAvoidance(3)
    env.reset()

    assert env.state[0] == 2
    assert env.state[1] == 1
    assert env.state[2] == 1


def test_sample_start_state_collision_avoidance():
    """tests :meth:`CollisionAvoidance.sample_start_state`"""
    env = CollisionAvoidance(7)
    np.testing.assert_array_equal(env.sample_start_state(), [6, 3, 3])


def test_step_collision_avoidance():
    """tests :meth:`CollisionAvoidance.step`"""

    env = CollisionAvoidance(7)

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

    up_env = CollisionAvoidance(7, (0, 0, 1))
    up_env.step(1)
    assert up_env.state[2] == 4
    up_env.step(1)
    assert up_env.state[2] == 5
    up_env.step(1)
    up_env.step(1)
    assert up_env.state[2] == 6

    down_env = CollisionAvoidance(7, (1, 0, 0))
    down_env.step(1)
    assert down_env.state[2] == 2
    down_env.step(1)
    assert down_env.state[2] == 1
    down_env.step(1)
    down_env.step(1)
    assert down_env.state[2] == 0


def test_utils_collision_avoidance():
    """Tests misc functionality in Collision Avoidance

    * :meth:`CollisionAvoidance.generate_observation`
    * :meth:`CollisionAvoidance.bound_in_grid`
    * :meth:`CollisionAvoidance.action_sace`
    * :meth:`CollisionAvoidance.observation_space`
    * :meth:`CollisionAvoidance.size`
    """

    env = CollisionAvoidance(7)

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


def setup_road_racer():
    """Returns a common setup for :class:`RoadRacer`"""
    length = 6
    num_lanes = int(np.random.choice(range(3, 15, 2)))
    probs = np.random.rand(num_lanes)
    random_env = RoadRacer(probs)

    return length, num_lanes, probs, random_env


def test_init_road_racer():
    """some tests on the :class:`RoadRacer` init

    raises on   * even lanes
                * probs outside of 0 and 1
                * less than 2 long

    """

    with pytest.raises(AssertionError):
        RoadRacer(np.array([0.5, 0.7]))
    with pytest.raises(AssertionError):
        RoadRacer(np.array([0.5, 0.7, 1.1]))
    with pytest.raises(AssertionError):
        RoadRacer(np.array([0.5, 0.7, -0.1]))


def test_properties_road_racer():
    """tests basic properties

    - :attr:`RoadRacer.num_lanes`
    - :attr:`RoadRacer.current_lane`
    - :attr:`RoadRacer.middle_lane`
    - :attr:`RoadRacer.action_space`
    - :attr:`RoadRacer.observation_space`
    - :attr:`RoadRacer.state_space`
    """

    length, num_lanes, _, random_env = setup_road_racer()

    # random domain properties
    assert random_env.lane_length == length
    assert random_env.num_lanes == num_lanes
    assert random_env.current_lane == (num_lanes - 1) / 2
    assert random_env.middle_lane == (num_lanes - 1) / 2

    assert random_env.action_space.n == 3

    obs_space = random_env.observation_space
    assert isinstance(obs_space, DiscreteSpace)
    np.testing.assert_array_equal(obs_space.size, [length])

    state_space = random_env.state_space
    assert isinstance(state_space, DiscreteSpace)
    np.testing.assert_array_equal(
        state_space.size,
        [length] * (num_lanes) + [num_lanes],
    )


def test_functional_road_racer():
    """tests the pure functions defined in the road racer package

    - :func:`road_racer.current_lane`
    - :func:`road_racer.get_observation`

    """

    _, _, _, random_env = setup_road_racer()

    state = np.random.randint(
        low=2,
        high=np.random.randint(low=3, high=5),
        size=np.random.randint(low=3, high=5),
    )

    state[-1] = 0
    assert RoadRacer.get_current_lane(state) == 0

    state[-1] = 3
    assert RoadRacer.get_current_lane(state) == 3

    state[-1] = -1
    with pytest.raises(AssertionError):
        RoadRacer.get_current_lane(state)

    dist = state[0]
    state[-1] = 0
    np.testing.assert_array_equal(random_env.get_observation(state), [dist])

    dist = state[1]
    state[-1] = 1
    np.testing.assert_array_equal(
        random_env.get_observation(state), [dist], f"state={state}"
    )


def test_reset_road_racer():
    """tests :meth:RoadRacer.reset`"""
    length, _, _, random_env = setup_road_racer()
    np.testing.assert_array_equal(random_env.reset(), [length - 1])


def test_start_start() -> None:
    """tests :meth:`RoadRacer.sample_start_state`"""

    length, num_lanes, _, random_env = setup_road_racer()
    start_state = np.ones(num_lanes + 1) * length - 1
    start_state[-1] = int(num_lanes / 2)

    np.testing.assert_array_equal(random_env.sample_start_state(), start_state)


def test_reward():
    """test :meth:`RoadRacer.reward`"""
    length, num_lanes, _, random_env = setup_road_racer()

    state = np.random.randint(low=2, high=length - 1, size=num_lanes + 1)

    # put agent in first lane
    state[-1] = 0

    next_state = random_env.simulation_step(state, RoadRacer.NO_OP).state

    dist = next_state[0]

    assert (
        random_env.reward(state, RoadRacer.NO_OP, new_state=next_state) == dist
    ), f"{state} -> {next_state}"

    # imagine agent tried to go up -> penalty of 1
    assert (
        random_env.reward(state, RoadRacer.GO_UP, new_state=next_state) == dist - 1
    ), f"{state} -> {next_state}"

    # imagine agent went down
    next_state[-1] = 1
    dist = next_state[1]
    assert (
        random_env.reward(state, RoadRacer.GO_DOWN, new_state=next_state) == dist
    ), f"{state} -> {next_state}"

    # imagine agent **tried** to go down but was not possible
    next_state[1] = 0
    next_state[-1] = 0
    dist = next_state[0]
    assert (
        random_env.reward(state, RoadRacer.GO_DOWN, new_state=next_state) == dist - 1
    ), f"{state} -> {next_state}"


def test_terminal():
    """tests :meth:`RoadRacer.terminal`"""
    length, num_lanes, _, random_env = setup_road_racer()

    # test any random <s,a,s'> transition
    assert not random_env.terminal(
        np.concatenate(
            (
                np.random.randint(low=1, high=length, size=num_lanes),
                [random.randint(0, num_lanes - 1)],
            )
        ),
        random_env.action_space.sample_as_int(),
        np.concatenate(
            (
                np.random.randint(low=1, high=length, size=num_lanes),
                [random.randint(0, num_lanes - 1)],
            )
        ),
    )


def test_steps():
    """tests :meth:`RoadRacer.step`"""
    length, num_lanes, _, random_env = setup_road_racer()

    go_up = RoadRacer.GO_UP
    stay = RoadRacer.NO_OP
    go_down = RoadRacer.GO_DOWN

    # test regular step: all lanes should either stay or advance one
    state = np.random.randint(low=2, high=length - 1, size=num_lanes + 1)
    state[-1] = 1

    next_state = random_env.simulation_step(state, stay).state
    for feature, value in enumerate(next_state):
        assert value in [
            state[feature],
            state[feature] - 1,
        ], f"{state} -> {next_state}, failing at feature {feature}"

    # test that lane advance p=1 will advance, and p=0 will stay
    random_env.lane_probs = np.ones(num_lanes)

    next_state = state - 1
    next_state[-1] = state[-1]

    np.testing.assert_array_equal(
        random_env.simulation_step(state, stay).state, next_state
    )

    random_env.lane_probs = np.zeros(num_lanes)
    np.testing.assert_array_equal(random_env.simulation_step(state, stay).state, state)

    # test changing lane
    state[-1] = 0
    assert random_env.simulation_step(state, go_down).state[-1] == 1

    state[-1] = num_lanes - 1
    assert random_env.simulation_step(state, go_up).state[-1] == num_lanes - 2

    state[-1] = 1
    assert random_env.simulation_step(state, go_down).state[-1] == 2
    assert random_env.simulation_step(state, go_up).state[-1] == 0

    # test changing lane on border will not change
    state[-1] = 0
    assert random_env.simulation_step(state, go_up).state[-1] == 0

    state[-1] = num_lanes - 1
    assert random_env.simulation_step(state, go_down).state[-1] == num_lanes - 1

    # test that changing lane into a car will not change lane
    state[-1] = 0
    assert random_env.simulation_step(state, go_down).state[-1] == 1

    state[-1] = num_lanes - 1
    assert random_env.simulation_step(state, go_up).state[-1] == num_lanes - 2

    state[-1] = 1
    assert random_env.simulation_step(state, go_down).state[-1] == 2
    assert random_env.simulation_step(state, go_up).state[-1] == 0

    # test cars starting in lane again after being passed
    random_env.lane_probs = np.ones(num_lanes)
    state[0] = 0
    state[-1] = 1
    np.testing.assert_array_equal(
        random_env.simulation_step(state, stay).state[0],
        length - 1,
    )

    # test lane will not advance if agent is blocking
    state[1] = 1
    np.testing.assert_array_equal(random_env.simulation_step(state, stay).state[1], 1)

    # test throwing terminal state if given state where agent is on car
    state[RoadRacer.get_current_lane(state)] = 0
    with pytest.raises(TerminalState):
        random_env.simulation_step(state, stay)


def test_road_race_prior_lane_probs():
    """some basic tests"""

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


def test_ca_prior_default():
    """tests the default prior"""
    # ignore private `_block_policy` usage
    # pylint: disable=W0212

    sampled_domain = CollisionAvoidancePrior(3, 1).sample()

    assert isinstance(sampled_domain, CollisionAvoidance)

    block_pol = sampled_domain._block_policy

    np.testing.assert_array_less(block_pol, 1)
    np.testing.assert_array_less(0, block_pol)

    assert round(abs(np.sum(block_pol) - 1), 7) == 0


def test_ca_prior_certain_prior():
    """very certain prior"""
    # ignore private `_block_policy` usage
    # pylint: disable=W0212
    sampled_domain = CollisionAvoidancePrior(3, 10000000).sample()

    assert isinstance(sampled_domain, CollisionAvoidance)

    block_pol = sampled_domain._block_policy

    assert round(abs(block_pol[0] - 0.05), 3) == 0
    assert round(abs(block_pol[1] - 0.9), 3) == 0
    assert round(abs(block_pol[2] - 0.05), 3) == 0


def test_ca_prior_uncertain_prior():
    """uncertain prior"""
    # ignore private `_block_policy` usage
    # pylint: disable=W0212
    sampled_domain = CollisionAvoidancePrior(3, 10).sample()

    assert isinstance(sampled_domain, CollisionAvoidance)

    block_pol = sampled_domain._block_policy

    assert round(abs(block_pol[0] - 0.05), 6) != 0
    assert round(abs(block_pol[1] - 0.9), 6) != 0
    assert round(abs(block_pol[2] - 0.05), 6) != 0


def test_encoding_gridworld_prior():
    """tests encoding is done correctly"""

    one_hot_sample = GridWorldPrior(size=3, one_hot_encode_goal=True).sample()
    assert one_hot_sample.observation_space.ndim == 5

    default_sample = GridWorldPrior(size=5, one_hot_encode_goal=False).sample()
    assert default_sample.observation_space.ndim == 3


def test_default_slow_cells_gridworld_prior():
    """tests Gridworlds sampled from prior have no slow cells"""

    sample_gridworld_1 = GridWorldPrior(size=5, one_hot_encode_goal=False).sample()
    sample_gridworld_2 = GridWorldPrior(size=5, one_hot_encode_goal=False).sample()

    assert isinstance(sample_gridworld_1, GridWorld)
    assert isinstance(sample_gridworld_2, GridWorld)

    assert sample_gridworld_1.slow_cells, "may **rarely** be empty"
    assert sample_gridworld_2.slow_cells, "may **rarely** be empty"
    assert (
        sample_gridworld_1.slow_cells != sample_gridworld_2.slow_cells
    ), "may **rarely** be true"


def test_encoding_tiger_prior():
    """tests the sample method encoding is correct"""

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


def test_observation_prob_tiger_prior():
    """tests the observation probability of samples"""
    # ignore private `_correct_obs_probs` usage
    # pylint: disable=W0212

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


def test_prior_correctness_tiger_prior():
    """tests the prior correctness parameter

    Values tested: .1, .5, .9999, 1.

    Args:

    RETURNS (`None`):

    """
    # ignore private `_correct_obs_probs` usage
    # pylint: disable=W0212
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


def test_num_total_counts_tiger_prior():
    """tests the parameter # of total counts

    Args:

    RETURNS (`None`):

    """
    # ignore private `_correct_obs_probs` usage
    # pylint: disable=W0212

    high_total_counts = 10000000.0
    low_total_counts = 1.0
    incorrect_prior_setting = 0.0

    encoding = random.choice([True, False])

    tiger = TigerPrior(high_total_counts, incorrect_prior_setting, encoding).sample()
    assert isinstance(tiger, Tiger)

    obs_probs = tiger._correct_obs_probs
    assert 0.624 <= obs_probs[0] <= 0.626
    assert 0.624 <= obs_probs[1] <= 0.626

    tiger = TigerPrior(low_total_counts, incorrect_prior_setting, encoding).sample()
    assert isinstance(tiger, Tiger)

    obs_probs = tiger._correct_obs_probs
    assert not (0.624 <= obs_probs[0] <= 0.626), f"Rarely false; obs = {obs_probs[0]}"
    assert not (0.624 <= obs_probs[1] <= 0.626), f"Rarely false; obs = {obs_probs[0]}"


if __name__ == "__main__":
    pytest.main([__file__])
