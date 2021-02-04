""" tests the functionality of priors """
import numpy as np
import pytest
import torch
from gym_gridverse.action import Action as GVerseAction
from gym_gridverse.envs.factory import env_from_descr
from gym_gridverse.envs.gridworld import GridWorld as GVerseGridworld
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)

from general_bayes_adaptive_pomdps.models.partial.domain.gridverse_gbapomdps import (
    GridversePositionAugmentedState,
    GridversePositionOrientationAugmentedState,
    agent_position,
    agent_position_and_orientation,
    create_gbapomdp,
    get_augmented_state_class,
    gverse_obs2array,
    noise_turn_orientation_transactions,
    open_backwards_positions,
    open_foward_positions,
    sample_state_with_random_agent,
    tnet_accuracy,
)


def test_get_augmented_state_class():
    assert get_augmented_state_class("position") == GridversePositionAugmentedState
    assert (
        get_augmented_state_class("position_and_orientation")
        == GridversePositionOrientationAugmentedState
    )

    with pytest.raises(ValueError):
        get_augmented_state_class("something else")


@pytest.mark.parametrize("pos", [((5, 2)), ((0, 0)), ((100, 50))])
def test_agent_position(pos):
    d = env_from_descr("KeyDoor-16x16-v0")
    s = d.functional_reset()
    s.agent.position = Position.from_position_or_tuple(pos)

    assert (agent_position(s) == pos).all()


@pytest.mark.parametrize("pos,orientation", [((5, 2), 0), ((0, 0), 2), ((100, 50), 3)])
def test_agent_position_and_orientation(pos, orientation):
    d = env_from_descr("KeyDoor-16x16-v0")
    s = d.functional_reset()
    s.agent.position = Position.from_position_or_tuple(pos)
    s.agent.orientation = Orientation(orientation)
    y, x, o = agent_position_and_orientation(s)

    assert y == pos[0]
    assert x == pos[1]
    assert o == orientation

    out = agent_position_and_orientation(s, one_hot_orientation=True)
    y = out[0]
    x = out[1]
    o = out[2:]
    assert out[0] == pos[0]
    assert x == pos[1]
    assert o.shape == (4,) and o.sum() == 1 and o[orientation] == 1


def test_gverse_obs2array():
    d = env_from_descr("Dynamic-Obstacles-6x6-v0")
    s = d.functional_reset()

    obs = gverse_obs2array(d, DefaultObservationRepresentation(d.observation_space), s)
    assert obs.shape == (297,)


@pytest.mark.parametrize(
    "height,width",
    [
        (3, 3),
        (10, 10),
        (7, 9),
    ],
)
def test_state_space(height, width):
    """Tests ``.state_space`` staticmethods of augmented states"""
    assert (
        GridversePositionAugmentedState.state_space(height, width).size
        == [
            height,
            width,
        ]
    ).all()
    assert (
        GridversePositionOrientationAugmentedState.state_space(height, width).size
        == [height, width, 4]
    ).all()


def test_state_input_size():
    """Tests ``.state_input_size`` staticmethods of augmented states"""
    assert GridversePositionAugmentedState.state_input_size() == 2
    assert GridversePositionOrientationAugmentedState.state_input_size() == 6


def model_equals(model_a, model_b) -> bool:
    """returns whether provided pytorch models are equal"""
    for tensor_a, tensor_b in zip(model_a.parameters(), model_b.parameters()):
        if not torch.equal(tensor_a.data, tensor_b.data):
            return False

    return True


# TODO: generalize to test "any" GBA-POMDP
@pytest.mark.parametrize(
    "model_type, expected_class",
    [
        ("position", GridversePositionAugmentedState),
        ("position_and_orientation", GridversePositionOrientationAugmentedState),
    ],
)
def test_gbapomdp(
    model_type,
    expected_class,
):
    d = env_from_descr("Dynamic-Obstacles-6x6-v0")
    assert isinstance(d, GVerseGridworld)

    gbapomdp = create_gbapomdp(
        d, "SGD", 0.1, 32, 0.0, 128, 8, 1, model_type, "", online_learning_rate=0.01
    )

    # test `action_space`
    assert gbapomdp.action_space.n == d.action_space.num_actions

    s = gbapomdp.sample_start_state()
    assert isinstance(s, expected_class)

    # testing `simulation_step`
    next_step = gbapomdp.simulation_step(s, 0, optimize=False)
    next_s1, obs = next_step.state, next_step.observation
    assert isinstance(s, expected_class)
    assert isinstance(next_s1, expected_class)
    assert not model_equals(s.learned_model.net, next_s1.learned_model.net)
    assert not s.domain_state == next_s1.domain_state

    next_s2 = gbapomdp.simulation_step(s, 0, optimize=True).state
    assert model_equals(s.learned_model.net, next_s2.learned_model.net)
    assert not s.domain_state == next_s1.domain_state

    # testing `domain_simulation_step`
    next_s = gbapomdp.domain_simulation_step(s, 1).state
    assert isinstance(next_s, expected_class)
    assert model_equals(s.learned_model.net, next_s.learned_model.net)
    assert not s.domain_state == next_s1.domain_state

    # testing `model_simulation_step`
    next_state = gbapomdp.model_simulation_step(s, s, 2, next_s1, obs)
    assert isinstance(next_state, expected_class)
    assert not model_equals(s.learned_model.net, next_state.learned_model.net)
    assert s.domain_state == next_state.domain_state

    next_state = gbapomdp.model_simulation_step(s, s, 2, next_s1, obs, optimize=True)
    assert isinstance(next_state, expected_class)
    assert model_equals(s.learned_model.net, next_state.learned_model.net)
    assert s.domain_state == next_state.domain_state


def test_sample_state_with_random_agent():
    d = env_from_descr("Empty-5x5-v0")

    states = [sample_state_with_random_agent(d) for _ in range(200)]
    positions = {s.agent.position.astuple() for s in states}
    orientations = {s.agent.orientation for s in states}

    assert len(positions) == 25
    assert len(orientations) == 4


def test_position_augmented_state():
    """Some random and basic tests on :class:`GridversePositionAugmentedState`"""

    # hacky way of getting a state
    d = env_from_descr("KeyDoor-16x16-v0")
    assert isinstance(d, GVerseGridworld)
    p = create_gbapomdp(
        d, "SGD", 0.01, 32, 0.0, 128, 8, 1, "position", "", 0.05
    ).sample_start_state

    s = p()
    assert isinstance(s, GridversePositionAugmentedState)

    # test updating model
    a = 3

    # test `domain_step`
    next_s, o = s.domain_step(a)
    assert isinstance(next_s, GridversePositionAugmentedState)
    assert not s.domain_state == next_s.domain_state
    assert model_equals(s.learned_model.net, next_s.learned_model.net)

    # test `update_model_distribution` and `model_accuracy`
    transition_data = [
        ((agent_position(s.domain_state), a), agent_position(next_s.domain_state))
    ]
    init_acc = list(tnet_accuracy(s.learned_model, transition_data))[0]

    s_with_updated_model = s.update_model_distribution(s, a, next_s, o, optimize=True)
    acc = list(tnet_accuracy(s.learned_model, transition_data))[0]
    assert isinstance(s_with_updated_model, GridversePositionAugmentedState)
    assert model_equals(s.learned_model.net, s_with_updated_model.learned_model.net)
    assert init_acc < acc

    s_with_updated_model = s.update_model_distribution(s, a, next_s, o)
    assert not model_equals(s.learned_model.net, s_with_updated_model.learned_model.net)
    assert acc == list(tnet_accuracy(s.learned_model, transition_data))[0]
    assert (
        acc
        < list(tnet_accuracy(s_with_updated_model.learned_model, transition_data))[0]
    )

    # test `model_accuracy`
    acc = np.array(list(s.model_accuracy(8)))

    assert acc.shape == (8,)
    assert (0 <= acc).all()
    assert (acc <= 1).all()


def test_position_and_orientation_augmented_state():
    """Some random and basic tests on :class:`GridversePositionOrientationAugmentedState`"""

    # hacky way of getting a state
    d = env_from_descr("KeyDoor-16x16-v0")
    assert isinstance(d, GVerseGridworld)
    p = create_gbapomdp(
        d,
        "SGD",
        0.1,
        32,
        0.0,
        128,
        8,
        1,
        model_type="position_and_orientation",
        prior_option="",
        online_learning_rate=0.05,
    ).sample_start_state

    s = p()
    assert isinstance(s, GridversePositionOrientationAugmentedState)

    # test updating model
    a = 3

    # test `domain_step`
    next_s, o = s.domain_step(a)
    assert isinstance(next_s, GridversePositionOrientationAugmentedState)
    assert not s.domain_state == next_s.domain_state
    assert model_equals(s.learned_model.net, next_s.learned_model.net)

    # test `update_model_distribution` and `model_accuracy`
    transition_data = [
        (
            (
                agent_position_and_orientation(
                    s.domain_state, one_hot_orientation=True
                ),
                a,
            ),
            agent_position_and_orientation(next_s.domain_state),
        )
    ]
    init_acc = list(tnet_accuracy(s.learned_model, transition_data))[0]

    s_with_updated_model = s.update_model_distribution(s, a, next_s, o, optimize=True)
    acc = list(tnet_accuracy(s.learned_model, transition_data))[0]
    assert isinstance(s_with_updated_model, GridversePositionOrientationAugmentedState)
    assert model_equals(s.learned_model.net, s_with_updated_model.learned_model.net)
    assert init_acc < acc

    s_with_updated_model = s.update_model_distribution(s, a, next_s, o)
    assert not model_equals(s.learned_model.net, s_with_updated_model.learned_model.net)
    assert acc == list(tnet_accuracy(s.learned_model, transition_data))[0]
    assert (
        acc
        < list(tnet_accuracy(s_with_updated_model.learned_model, transition_data))[0]
    )

    # test `model_accuracy`
    acc = np.array(list(s.model_accuracy(8)))

    assert acc.shape == (8,)
    assert (0 <= acc).all()
    assert (acc <= 1).all()


def test_noise_turn_orientation_transactions():
    """tests :func:`noise_turn_orientation_transactions`"""
    d = env_from_descr("Empty-5x5-v0")
    assert isinstance(d, GVerseGridworld)

    states, actions, next_states = noise_turn_orientation_transactions(d, 32)

    atleast_one_transition_has_wrong_orientation = False
    for s, a, ss in zip(states, actions, next_states):
        real_ss = d.functional_step(s, d.action_space.int_to_action(a))[0]

        if (
            d.action_space.int_to_action(a) == GVerseAction.TURN_LEFT
            or d.action_space.int_to_action(a) == GVerseAction.TURN_RIGHT
        ):
            if real_ss.agent.orientation != ss.agent.orientation:
                atleast_one_transition_has_wrong_orientation = True
        else:  # make sure that for normal moves we have the correct orientation
            assert real_ss.agent.orientation == ss.agent.orientation

    assert atleast_one_transition_has_wrong_orientation


@pytest.mark.parametrize(
    "pos,o,forward_positions,max_dist",
    [
        ((0, 0), Orientation.N, [], None),
        ((1, 1), Orientation.N, [(1, 1)], None),
        ((1, 1), Orientation.W, [(1, 1)], None),
        ((2, 2), Orientation.W, [(2, 2), (2, 1)], None),
        ((2, 2), Orientation.N, [(2, 2), (1, 2)], None),
        ((2, 2), Orientation.E, [(2, 2), (2, 3), (2, 4), (2, 5)], None),
        ((2, 2), Orientation.E, [(2, 2), (2, 3), (2, 4), (2, 5)], 4),
        ((2, 2), Orientation.E, [(2, 2), (2, 3), (2, 4), (2, 5)], 3),
        ((2, 2), Orientation.E, [(2, 2), (2, 3), (2, 4)], 2),
        ((2, 2), Orientation.E, [(2, 2), (2, 3)], 1),
        ((2, 2), Orientation.E, [(2, 2)], 0),
    ],
)
def test_open_forward_positions(pos, o, forward_positions, max_dist):
    s = env_from_descr("Empty-5x5-v0").functional_reset()
    assert list(open_foward_positions(s, pos, o, max_dist)) == forward_positions


@pytest.mark.parametrize(
    "pos,o,backwards_positions,max_dist",
    [
        ((0, 0), Orientation.S, [], None),
        ((1, 1), Orientation.S, [(1, 1)], None),
        ((1, 1), Orientation.E, [(1, 1)], None),
        ((2, 2), Orientation.E, [(2, 2), (2, 1)], None),
        ((2, 2), Orientation.S, [(2, 2), (1, 2)], None),
        ((2, 2), Orientation.W, [(2, 2), (2, 3), (2, 4), (2, 5)], None),
        ((2, 2), Orientation.W, [(2, 2), (2, 3), (2, 4), (2, 5)], 4),
        ((2, 2), Orientation.W, [(2, 2), (2, 3), (2, 4), (2, 5)], 3),
        ((2, 2), Orientation.W, [(2, 2), (2, 3), (2, 4)], 2),
        ((2, 2), Orientation.W, [(2, 2), (2, 3)], 1),
        ((2, 2), Orientation.W, [(2, 2)], 0),
    ],
)
def test_open_backwards_positions(pos, o, backwards_positions, max_dist):
    s = env_from_descr("Empty-5x5-v0").functional_reset()
    assert list(open_backwards_positions(s, pos, o, max_dist)) == backwards_positions
