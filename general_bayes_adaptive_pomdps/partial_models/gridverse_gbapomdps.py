"""Contains partial GBA-POMDP implementations for the gridverse domain

The [gridverse domain](https://github.com/abaisero/gym-gridverse) is a very
interesting problem. Unfortunately it is also quite complicated and large
(depending on the instance). Hence, we simplify the task by assuming/setting
some parts of the dynamics known. We do so by instantiating partial GBA-POMDPs.

The general approach is to have augmented states contain:

    - an actual domain state (complex object)
    - a model that predicts a subset of the domain state
    - prior knowledge (gridverse dynamics)

For example, we can try to learn to predict the agent's location, provided that
the other part of the dynamics (observation model, 'dynamics of walls') are
known. In that case, the dynamics of the GBA-POMDP would be as follows::

    next_pos = model(domain_state.agent_pos, incoming_action)
    next_domain_state = real_gridverse_dynamics(domain_state, incoming_action)
    next_domain_state.agent_pos = next_pos
    obs = real_gridverse_dynamics(domain_state, incoming_action, next_domain_state)

    model.update(domain_state, incoming_action, next_domain_state, obs)

    return (next_domain_state, model, obs)

Here we are "cheating" in the sense that the real dynamics of gridverse are
_used_. This makes the GBA-POMDP "partial". However, by setting a subset of the
values in ``next_domain_state`` to those predicted by our ``model``, we emulate
learning with prior knowledge.
"""

import itertools as itt
import logging
import math
import random
from copy import deepcopy
from functools import partial
from typing import Callable, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import torch
from cached_property import cached_property
from gym_gridverse.action import ROTATION_ACTIONS as GVERSE_ROTATION_ACTIONS
from gym_gridverse.action import Action as GVerseAction
from gym_gridverse.envs.gridworld import GridWorld as GVerseGridworld
from gym_gridverse.envs.inner_env import InnerEnv as GVerseEnv
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)
from gym_gridverse.representations.representation import ObservationRepresentation
from gym_gridverse.state import State as GVerseState

from general_bayes_adaptive_pomdps.core import ActionSpace
from general_bayes_adaptive_pomdps.misc import DiscreteSpace
from general_bayes_adaptive_pomdps.models.neural_networks.misc import whiten_input
from general_bayes_adaptive_pomdps.models.neural_networks.neural_pomdps import (
    DynamicsModel,
    get_optimizer_builder,
)
from general_bayes_adaptive_pomdps.models.neural_networks.pytorch_api import device
from general_bayes_adaptive_pomdps.partial_models.partial_gbapomdp import (
    AugmentedGodState,
    GBAPOMDPThroughAugmentedState,
)


class GridverseAugmentedGodState(AugmentedGodState, Protocol):
    """A (protocol) class that brings together all grid-verse augmented states

    I noticed a pattern in the implementation of these classes and a need to
    provide this abstraction. This is mainly used to allow generalization, such
    as through the typing system.

    It defines some functions and property that all augmented states provide.
    """

    @staticmethod
    def state_input_size() -> int:
        """The dimensionality of the state as input"""

    @staticmethod
    def state_space(height: int, width: int) -> DiscreteSpace:
        """The state space of the underlying POMDP that is being learned

        Note that this is not the state space of the entire GBA-POMDP, nor the
        state space of the entire underlying POMDP. This is the state space of
        the part of the model that is (learned to) predict.
        """

    @staticmethod
    def train_tnet(
        net: DynamicsModel.TNet,
        states: List[GVerseState],
        actions: List[int],
        next_states: List[GVerseState],
    ) -> float:
        """Train ``net`` on batch of data

        :param net: model to be trained
        :param states: states at t
        :param actions: taken actions at t
        :param next_states: states at t+1
        :returns: loss on training
        """

    @staticmethod
    def tnet_accuracy(
        net: DynamicsModel.TNet, domain: GVerseGridworld, num_samples: int
    ) -> Iterable[float]:
        """Returns a set of ``num_samples`` accuracy scores

        Approximates the probability of correctly predicting the position
        transition with ``n`` samples.

        :param net: net to be tested on accuracy
        :param domain: current real domain (used to generate data measure with)
        :param num_samples: number of (MC) samples
        :returns: probabilities of sampling correct output
        """


def train_models(
    models: List[DynamicsModel.TNet],
    num_pretrain_epochs: int,
    data_sampler: Callable[[], Tuple[List[GVerseState], List[int], List[GVerseState]]],
) -> List[DynamicsModel.TNet]:
    """Trains ``models`` on data from ``data_sampler``

    :param models: the transition networks to be trained
    :param num_pretrain_epochs: number of batches to train each model on
    :param data_sampler: a function to draw data from
    :returns: the models trained, in case you want to use that (modifies input ``models``)
    """

    logger = logging.getLogger("create gridverse prior")

    for i, net in enumerate(models):
        logger.debug("Training net %s / %s...", i + 1, len(models))

        for _ in range(num_pretrain_epochs):

            states, actions, next_states = data_sampler()

            GridversePositionOrientationAugmentedState.train_tnet(
                net,
                states,
                actions,
                next_states,
            )

    return models


def agent_position(state: GVerseState) -> np.ndarray:
    """Get the agent position as numpy array from state

    :param state: the grid-verse state that contains the agent position
    :returns: [y, x]
    """
    return np.array(state.agent.position.astuple())


def agent_position_and_orientation(
    state: GVerseState, one_hot_orientation: bool = False
) -> np.ndarray:
    """Get the agent position as numpy array from state

    :param state: the grid-verse state that contains the agent position
    :returns: [y, x]
    """
    if one_hot_orientation:
        orientation = np.zeros(len(Orientation), dtype=int)
        orientation[state.agent.orientation.value] = 1
    else:
        orientation = [state.agent.orientation.value]

    return np.concatenate([agent_position(state), orientation])


def gverse_obs2array(
    domain: GVerseEnv, obs_rep: ObservationRepresentation, s: GVerseState
) -> np.ndarray:
    """Converts a state in the Gridworld to a numpy observation"""
    dict_of_obs = obs_rep.convert(domain.functional_observation(s))
    return np.concatenate([obs.flatten() for obs in dict_of_obs.values()])


def sample_state_with_random_agent(d: GVerseEnv) -> GVerseState:
    """Samples a state in ``d`` with a random position and orientation

    :param d: the domain to sample states from
    :returns: (initial) state but with random position and orientation
    """
    s = d.functional_reset()
    random_open_position = random.choice(
        [p for p in s.grid.positions() if not s.grid[p].blocks]
    )
    random_orientation = random.choice(list(Orientation))

    s.agent.position = random_open_position
    s.agent.orientation = random_orientation

    return s


def sample_random_interaction(
    d: GVerseGridworld,
) -> Tuple[GVerseState, GVerseAction, GVerseState]:
    """samples a random interaction in ``d``

    #. samples a state from :func:`sample_state_with_random_agent`
    #. samples an action from ``d``
    #. samples a next state d(s,a)

    :param d: domain in which the interaction takes place
    :returns: a (state, action, state) tuple
    """
    s = sample_state_with_random_agent(d)
    a = random.choice(d.action_space.actions)
    next_s = d.functional_step(s, a)[0]

    return s, a, next_s


def tnet_accuracy(net: DynamicsModel.TNet, test_data) -> Iterable[float]:
    """Returns a generator of accuracy

    Approximates the probability of correctly predicting  on ``test_data``

    :param net:
    :param test_data: generator of ((state, action), next_state) data
    :rtype: Iterable[float]
    """

    # sequence of [p(y), y], where y is a feature vector
    predictions = ((net.model(s, a), ss) for (s, a), ss in test_data)

    # probability of label: multiplication of probs of each feature
    return (
        math.prod([dist[x] for dist, x in zip(model, ss)]) for model, ss in predictions
    )


class GridversePositionOrientationAugmentedState(GridverseAugmentedGodState):
    """An augmented class that creates a partial GBA-POMDP for prediction position

    This class, when used in the :class:`GBAPOMDPThroughAugmentedState`,
    creates a partial GBA-POMDP that predicts the position and orientation of
    the agent in the state dynamics. The other parts of the model are given.

    """

    def __init__(
        self,
        initial_state: GVerseState,
        learned_model: DynamicsModel.TNet,
        pomdp: GVerseGridworld,
        obs_rep: Callable[[GVerseState], np.ndarray],
    ):
        self.domain_state = initial_state
        self.learned_model = learned_model
        self.pomdp = pomdp
        self._obs_rep = obs_rep

    @staticmethod
    def state_input_size() -> int:
        """Part of protocol of :class:`GridverseAugmentedGodState`"""
        return 2 + len(Orientation)  # 1-hot encoding of orientation

    @staticmethod
    def state_space(height: int, width: int) -> DiscreteSpace:
        """Part of protocol of :class:`GridverseAugmentedGodState`"""
        return DiscreteSpace([height, width, len(Orientation)])

    def update_model_distribution(
        self,
        state: AugmentedGodState,
        action: int,
        next_state: AugmentedGodState,
        obs: np.ndarray,
        optimize: bool = False,
    ) -> "GridversePositionOrientationAugmentedState":
        """Updates the model distribution parameters with (s,a,s',o) data

        Given the prior over the model in ``self``, computes a new posterior
        given the (``state``, ``action``, ``next_state``, ``obs``)
        transition.

        Part of protocol of :class:`AugmentedGodState`. Updates the POMDP state
        according to the model in ``self``.

        NOTE: modifies ``self`` if ``optimize`` is true

        The updates applied are set during initialization.

        :param state: state at t
        :param action: action at t
        :param next_state: state at t+1
        :param obs: ignored
        :param optimize: whether to update model from ``self``
        """
        assert isinstance(state, GridversePositionOrientationAugmentedState)
        assert isinstance(next_state, GridversePositionOrientationAugmentedState)

        net_to_update = self.learned_model if optimize else deepcopy(self.learned_model)

        GridversePositionOrientationAugmentedState.train_tnet(
            net_to_update,
            [state.domain_state],
            [action],
            [next_state.domain_state],
        )

        return GridversePositionOrientationAugmentedState(
            self.domain_state, net_to_update, self.pomdp, self._obs_rep
        )

    def domain_step(self, action: int) -> Tuple["AugmentedGodState", np.ndarray]:
        """Applies ``action`` on POMDP state in ``self``

        Part of protocol of :class:`AugmentedGodState`. Updates the POMDP state according to the model in ``self``:

            #. samples the next position according to learned model
            #. samples next state according to POMDP dynamics
            #. sets next position to the sampled one
            #. returns observation by known model

        :param action: action taken by agent
        :returns: updated augmented state and generated observation
        """

        # (learned) prediction part
        next_y, next_x, next_orientation = self.learned_model.sample(
            self.domain_state_to_network_input(self.domain_state),
            action,
            num=1,
        )

        # known-part
        next_domain_state = self.pomdp.functional_step(
            self.domain_state, self.pomdp.action_space.int_to_action(action)
        )[0]

        # merge
        next_domain_state.agent.position = Position(next_y, next_x)
        next_domain_state.agent.orientation = Orientation(next_orientation)

        next_state = GridversePositionOrientationAugmentedState(
            next_domain_state, self.learned_model, self.pomdp, self._obs_rep
        )

        obs = next_state.observation

        return next_state, obs

    def reward(self, action: int, next_state: AugmentedGodState) -> float:
        assert isinstance(next_state, GridversePositionOrientationAugmentedState)
        return self.pomdp.reward_function(
            self.domain_state,
            self.pomdp.action_space.int_to_action(action),
            next_state.domain_state,
        )

    def terminal(self, action: int, next_state: AugmentedGodState) -> bool:
        assert isinstance(next_state, GridversePositionOrientationAugmentedState)
        return self.pomdp.termination_function(
            self.domain_state,
            self.pomdp.action_space.int_to_action(action),
            next_state.domain_state,
        )

    @cached_property
    def observation(self) -> np.ndarray:
        return self._obs_rep(self.domain_state)

    def model_accuracy(self, n: int = 8) -> List[float]:
        """Returns the model accuracy

        Returns ``n`` accuracy values that represent how well the model
        predicts the position. The initial position and action is sampled
        randomly (``n`` times), and the probability of predicting the correct
        transition is returned.

        :param n: number of accuracy samples
        :returns: list of accuracy [0,1]
        """
        return list(self.tnet_accuracy(self.learned_model, self.pomdp, n))

    @staticmethod
    def tnet_accuracy(
        net: DynamicsModel.TNet, domain: GVerseGridworld, num_samples: int
    ) -> Iterable[float]:
        """Returns a set of ``num_samples`` accuracy scores

        Approximates the probability of correctly predicting the position
        transition with ``n`` samples.

        Part of :class:`GridverseAugmentedGodState` protocol

        :param net: net to be tested on accuracy
        :param domain: current real domain (used to generate data measure with)
        :param num_samples: number of (MC) samples
        :returns: probabilities of sampling correct output
        """
        # create (s,a,s) data
        interactions = (sample_random_interaction(domain) for _ in range(num_samples))

        # reshape to correct input/output format for pos_TNet (position only)
        test_data = (
            (
                (
                    GridversePositionOrientationAugmentedState.domain_state_to_network_input(
                        s
                    ),
                    domain.action_space.action_to_int(a),
                ),
                agent_position_and_orientation(next_s),
            )
            for (s, a, next_s) in interactions
        )

        return tnet_accuracy(net, test_data)

    @staticmethod
    def train_tnet(
        net: DynamicsModel.TNet,
        states: List[GVerseState],
        actions: List[int],
        next_states: List[GVerseState],
    ) -> float:
        """Train ``net`` on batch of data

        Part of :class:`GridverseAugmentedGodState` protocol

        :param net: model to be trained
        :param states: states at t
        :param actions: taken actions at t
        :param next_states: states at t+1
        :returns: loss on training
        """
        a = torch.eye(net.action_space.n)[actions]
        state_input_rep = torch.FloatTensor(
            [
                GridversePositionOrientationAugmentedState.domain_state_to_network_input(
                    s
                )
                for s in states
            ]
        ).to(device())
        state_output_rep = torch.LongTensor(
            [agent_position_and_orientation(next_s) for next_s in next_states]
        ).to(device())

        return net.batch_train(state_input_rep, a, state_output_rep)

    @staticmethod
    def domain_state_to_network_input(s: GVerseState) -> np.ndarray:
        """Converts the domain into a nump array

        Returns a (6,) array, where the first two elements are the y and x
        value, whitened, and the others is a one-hot encoding of the
        orientation. Whitened here means normalized between -1 and 1

        :param s:
        :returns: [y, x, one-hot(orientation)] with y/x whitened if asked to
        """
        pos_and_orientation = agent_position_and_orientation(
            s, one_hot_orientation=True
        ).astype(float)

        pos_and_orientation[:2] = whiten_input(
            pos_and_orientation[:2],
            np.array([s.grid.shape.height, s.grid.shape.width]),
        )

        return pos_and_orientation


def uniform_true_transactions(
    domain: GVerseGridworld,
    batch_size: int,
) -> Tuple[List[GVerseState], List[int], List[GVerseState]]:
    """Generates transitions truly uniform according to true dynamics

    #. samples _initial_ state
    #. set position and orientation randomly
    #. sample random action
    #. step in domain

    :param domain:
    :param batch_size: length of returned lists
    :returns: lists of states, action, next states
    """

    transitions = (sample_random_interaction(domain) for _ in range(batch_size))
    states, gridverse_actions, next_states = zip(*transitions)
    actions = [domain.action_space.action_to_int(a) for a in gridverse_actions]

    return states, actions, next_states


def noise_turn_orientation_transactions(
    domain: GVerseGridworld,
    batch_size: int,
) -> Tuple[List[GVerseState], List[int], List[GVerseState]]:
    """Generates transitions where the orientation after turning is uniformly sampled

    Otherwise the data follows the true dynamics

    #. samples _initial_ state
    #. set position and orientation randomly
    #. sample random action
    #. step in domain
    #. if action is turning, set orientation randomly

    :param domain:
    :param batch_size: length of returned lists
    :returns: lists of states, action, next states
    """
    turn_actions = [
        domain.action_space.action_to_int(rot_act)
        for rot_act in GVERSE_ROTATION_ACTIONS
    ]

    states, actions, next_states = uniform_true_transactions(domain, batch_size)

    for a, next_s in zip(actions, next_states):
        if a in turn_actions:
            next_s.agent.orientation = random.choice(list(Orientation))

    return states, actions, next_states


def open_backwards_positions(
    s: GVerseState, pos: Position, o: Orientation, max_dist: Optional[int] = None
) -> Iterable[Position]:
    """Returns all open positions in (straight line) behind the agent

    Returns an iterator of positions, one step at a time forward relative to
    ``pos`` given ``o``, until (excluding) one blocks.

    :param s: current state
    :param pos: position of the agent
    :param o: orientation of the agent
    :param max: maximum distance in front of agent (infinite if not given)
    :returns: a generator of positions that are not blocked
    """
    delta_pos = o.as_position()

    it = itt.count() if max_dist is None else range(max_dist + 1)

    candidate_forward_positions = (
        pos + Position(delta_pos.y * -i, delta_pos.x * -i) for i in it
    )
    possible_positions = itt.takewhile(
        lambda pos: not s.grid[pos].blocks, candidate_forward_positions
    )

    return possible_positions


def open_foward_positions(
    s: GVerseState, pos: Position, o: Orientation, max_dist: Optional[int] = None
) -> Iterable[Position]:
    """Returns all open positions in (straight line) front of agent

    Returns an iterator of positions, one step at a time forward relative to
    ``pos`` given ``o``, until (excluding) one blocks.

    :param s: current state
    :param pos: position of the agent
    :param o: orientation of the agent
    :param max: maximum distance in front of agent (infinite if not given)
    :returns: a generator of positions that are not blocked
    """
    delta_pos = o.as_position()

    it = itt.count() if max_dist is None else range(max_dist + 1)

    candidate_forward_positions = (
        pos + Position(delta_pos.y * i, delta_pos.x * i) for i in it
    )
    possible_positions = itt.takewhile(
        lambda pos: not s.grid[pos].blocks, candidate_forward_positions
    )

    return possible_positions


def noise_foward_transitions(
    domain: GVerseGridworld,
    batch_size: int,
) -> Tuple[List[GVerseState], List[int], List[GVerseState]]:
    """Generates transitions where the forward actions moves arbitrary far (and 1 step back)

    Otherwise the data follows the true dynamics

    #. samples _initial_ state
    #. set position and orientation randomly
    #. sample random action
    #. step in domain
    #. if action is forward, move any forward position that is not blocked

    :param domain:
    :param batch_size: length of returned lists
    :returns: lists of states, action, next states
    """
    states, actions, next_states = uniform_true_transactions(domain, batch_size)

    for s, a, next_s in zip(states, actions, next_states):
        if a == domain.action_space.action_to_int(GVerseAction.MOVE_FORWARD):
            backwards_positions = list(
                open_backwards_positions(
                    next_s, s.agent.position, s.agent.orientation, max_dist=1
                )
            )
            forward_positions = list(
                open_foward_positions(
                    next_s, s.agent.position, s.agent.orientation, max_dist=5
                )
            )
            next_s.agent.position = random.choice(
                list(set(backwards_positions + forward_positions))
            )

    return states, actions, next_states


def create_data_sampler(
    domain: GVerseGridworld, batch_size: int, option: str
) -> Callable[[], Tuple[List[GVerseState], List[int], List[GVerseState]]]:
    """Creates or picks a grid-verse data generator

    Based on ``option`` this can either return
    :func:`uniform_true_transactions`, data from the true domain, or some
    more noisy data generator.

    :param domain:
    :param batch_size:
    :param option: type of prior ["", "noise_turn_orientation"]
    :returns: A callable to generate transition data
    :raises ValueError: if ``option`` not valid
    """
    if option == "":
        return partial(uniform_true_transactions, domain, batch_size)
    if option == "noise_turn_orientation":
        return partial(noise_turn_orientation_transactions, domain, batch_size)
    if option == "noise_forward_step":
        return partial(noise_foward_transitions, domain, batch_size)

    raise ValueError(
        f"{option} not in ['', 'noise_turn_orientation', 'noise_foward_step']"
    )


def create_gbapomdp(
    domain: GVerseGridworld,
    optimizer_name: str,
    learning_rate: float,
    network_size: int,
    dropout_rate: float,
    num_pretrain_epochs: int,
    batch_size: int,
    num_nets: int,
    prior_option: str,
    online_learning_rate: Optional[float],
) -> GBAPOMDPThroughAugmentedState:
    """Creates the gridverse GBA-POMDP

    The GBA-POMDP for the grid-verse problem depends mostly on the prior. Once
    a prior is given, which determines the type of augmented state, the rest
    follows. Hence most of the configurations depends on how to train and
    determine the prior states.

    This function returns a GBA-PODMP, based on the
    :class:`GBAPOMDPThroughAugmentedState`. The prior pre-trains models, given
    the ``optimizer_name``, ``learning_rate``, ``network_size``.
    ``dropout_rate``, ``num_pretrain_epochs`` and ``batch_size``.

    The ``prior_option`` determines on what data the models are pre-trained.:

        - "": uniform random transitions in the true enviornment
        - "noise_turn_orientation": "", but the orientation after turning is uniform
        - "noise_foward_step": "", but the forward step is :func:`noise_foward_transitions`

    :param domain: the underlying grid-verse domain
    :param optimizer_name: which optimizer to train with ["SGD", "Adam"]
    :param learning_rate: learning rate to use for training
    :param network_size: size of networks (2-layer, this determines # nodes)
    :param dropout_rate: how much nodes to drop during a forward-pass
    :param num_pretrain_epochs: number of batches to train on for the prior
    :param batch_size: size of a batch during pre-training
    :param num_nets: number of networks to train in the prior phase
    :param prior_option: type of prior ["", "noise_turn_orientation", "noise_foward_step"]
    :param online_learning_rate: (optional) learning rate during online steps
    :returns: GBAPOMDPThroughAugmentedState GBA-POMDP for grid-verse
    """

    assert 0 <= learning_rate <= 1
    assert network_size > 0
    assert 0 <= dropout_rate <= 1
    assert num_pretrain_epochs >= 0
    assert batch_size > 0
    assert num_nets > 0

    optimizer_builder = get_optimizer_builder(optimizer_name)
    action_space = ActionSpace(domain.action_space.num_actions)
    data_sampler = create_data_sampler(domain, batch_size, prior_option)
    obs_space = None  # XXX: could be useful at some point

    obs_rep = partial(
        gverse_obs2array,
        domain,
        DefaultObservationRepresentation(domain.observation_space),
    )

    # XXX: ugly
    grid_shape = domain.functional_reset().grid.shape
    h, w = grid_shape.height, grid_shape.width

    models = [
        DynamicsModel.TNet(
            GridversePositionOrientationAugmentedState.state_space(h, w),
            action_space,
            optimizer_builder,
            learning_rate,
            network_size,
            dropout_rate,
            input_state_size=GridversePositionOrientationAugmentedState.state_input_size(),
        )
        for _ in range(num_nets)
    ]

    train_models(
        models,
        num_pretrain_epochs,
        data_sampler,
    )

    if online_learning_rate is not None:
        assert 0 <= online_learning_rate < 1.0
        for m in models:
            m.set_learning_rate(online_learning_rate)

    def prior() -> GridversePositionOrientationAugmentedState:
        return GridversePositionOrientationAugmentedState(  # type: ignore
            domain.functional_reset(),
            random.choice(deepcopy(models)),
            domain,
            obs_rep,
        )

    return GBAPOMDPThroughAugmentedState(prior, action_space, obs_space)
