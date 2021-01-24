"""Contains partial GBA-POMDP implementations for the gridverse environment

The [gridverse environment](https://github.com/abaisero/gym-gridverse) is a
very interesting environment. Unfortunately it is also quite complicated and
large (depending on the instance). Hence, we simplify the task by
assuming/setting some parts of the dynamics known. We do so by instantiating
partial GBA-POMDPs.

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
import logging
import random
from copy import deepcopy
from functools import partial
from typing import Callable, List

import numpy as np
import torch
from gym_gridverse.action import Action as GVerseAction
from gym_gridverse.envs.gridworld import GridWorld as GVerseGridworld
from gym_gridverse.envs.inner_env import InnerEnv as GVerseEnv
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)
from gym_gridverse.representations.representation import ObservationRepresentation
from gym_gridverse.state import State as GVerseState

from general_bayes_adaptive_pomdps.agents.neural_networks.neural_pomdps import (
    DynamicsModel,
    get_optimizer_builder,
)
from general_bayes_adaptive_pomdps.environments import ActionSpace
from general_bayes_adaptive_pomdps.misc import DiscreteSpace
from general_bayes_adaptive_pomdps.models.partial.partial_gbapomdp import (
    AugmentedGodState,
    Prior,
)
from general_bayes_adaptive_pomdps.pytorch_api import (
    device,
    log_tensorboard,
    tensorboard_logging,
)


def create_gridverse_prior(
    domain: GVerseGridworld,
    optimizer_name: str,
    learning_rate: float,
    network_size: int,
    dropout_rate: float,
    num_pretrain_epochs: int,
    batch_size: int,
) -> Prior:
    """Given a gridworld, this functions sets up a prior for the GBA-POMDP

    Creates a prior for the :class:`general_bayes_adaptive_pomdps.models.partial.partial_gbapomdp`

    TODO: generalize to different representations

    :param domain: the gridverse gridworld we are building a GBA-POMDP out of
    :param optimizer_name:
    :param learning_rate:
    :param network_size:
    :param dropout_rate:
    :param num_pretrain_epochs:
    :param batch_size:
    :return: a prior to sample initial states from
    """
    assert 0 <= learning_rate <= 1
    assert network_size > 0
    assert 0 <= dropout_rate <= 1
    assert num_pretrain_epochs >= 0
    assert batch_size > 1

    logger = logging.getLogger("create gridverse prior")

    optimizer_builder = get_optimizer_builder(optimizer_name)

    grid_shape = domain.functional_reset().grid.shape
    h, w = grid_shape.height, grid_shape.width

    action_space = ActionSpace(domain.action_space.num_actions)

    # specific to this partial model: learn model over position
    pos_space = DiscreteSpace([h, w])
    models = [
        DynamicsModel.TNet(
            pos_space,
            action_space,
            optimizer_builder,
            learning_rate,
            network_size,
            dropout_rate,
        )
    ]

    # TODO: extract
    tboard_logging = tensorboard_logging()
    for i, net in enumerate(models):
        logger.debug("Training net %s / %s...", i, len(models))
        for batch in range(num_pretrain_epochs):
            # sample transitions
            states = [domain.functional_reset() for _ in range(batch_size)]

            # set position and orientation randomly
            for s in states:
                s.agent.position = Position(
                    random.randint(0, h - 1), random.randint(0, w - 1)
                )
                s.agent.orientation = random.choice(list(Orientation))

            actions = [
                random.randint(0, domain.action_space.num_actions - 1)
                for _ in range(batch_size)
            ]

            next_states = [
                domain.functional_step(s, GVerseAction(a))[0]
                for s, a in zip(states, actions)
            ]

            loss = train_TNet_on_position(
                net, states, actions, next_states, domain.action_space.num_actions
            )

            if tboard_logging:
                log_tensorboard(f"transition_loss/{net}", loss, batch)

    def prior() -> GridversePositionAugmentedState:
        return GridversePositionAugmentedState(
            domain.functional_reset(),
            random.choice(deepcopy(models)),
            domain,
            partial(
                gverse_obs2array,
                domain,
                DefaultObservationRepresentation(domain.observation_space),
            ),
        )

    return prior


def agent_position(state: GVerseState) -> np.ndarray:
    """Get the agent position as numpy array from state

    :param state: the grid-verse state that contains the agent position
    :returns: [x, y]
    """
    return np.array(state.agent.position.astuple())


def gverse_obs2array(
    domain: GVerseEnv, obs_rep: ObservationRepresentation, s: GVerseState
) -> np.ndarray:
    """Converts a state in the Gridworld to a numpy observation"""
    dict_of_obs = obs_rep.convert(domain.functional_observation(s))
    return np.concatenate([obs.flatten() for obs in dict_of_obs.values()])


def train_TNet_on_position(
    net: DynamicsModel.TNet,
    states: List[GVerseState],
    actions: List[int],
    next_states: List[GVerseState],
    num_actions: int,
) -> float:
    """Train ``net`` on batch of data

    :param net: model to be trained
    :param states: states at t
    :param actions: taken actions at t
    :param next_states: states at t+1
    :param num_actions: total number of possible actions (required for one-hot encoding)
    :return: loss on training
    """

    a = torch.eye(num_actions)[actions]
    pos = torch.FloatTensor([agent_position(s) for s in states]).to(device())
    next_pos = torch.LongTensor([agent_position(next_s) for next_s in next_states]).to(
        device()
    )

    return net.batch_train(pos, a, next_pos)


class GridversePositionAugmentedState(AugmentedGodState):
    """An augmented class that creates a partial GBA-POMDP for prediction position

    This class, when used in the :class:`GBAPOMDPThroughAugmentedState`,
    creates a partial GBA-POMDP that predicts only the position of the agent in
    the state dynamics. The other parts of the model are given.

    TODO: generalize how to update the model

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

    def update_theta(
        self,
        state: AugmentedGodState,
        action: int,
        next_state: AugmentedGodState,
        obs: np.ndarray,
    ):
        """Updates the model distribution parameters with (s,a,s',o) data

        Given the prior over the model in ``self``, computes a new posterior
        given the (``state``, ``action``, ``next_state``, ``obs``)
        transition.

        Part of protocol of :class:`AugmentedGodState`. Updates the POMDP state
        according to the model in ``self``.

        NOTE: modifies ``self``

        The updates applied are set during initialization.

        :param state: state at t
        :param action: action at t
        :param next_state: state at t+1
        :param obs: state at t+1
        """
        assert isinstance(state, GridversePositionAugmentedState)
        assert isinstance(next_state, GridversePositionAugmentedState)
        train_TNet_on_position(
            self.learned_model,
            [state.domain_state],
            [action],
            [next_state.domain_state],
            self.pomdp.action_space.num_actions,
        )

    def update_domain_state(self, action: int) -> np.ndarray:
        """Applies ``action`` on POMDP state in ``self``

        Part of protocol of :class:`AugmentedGodState`. Updates the POMDP state according to the model in ``self``:

            #. samples the next position according to learned model
            #. samples next state according to POMDP dynamics
            #. sets next position to the sampled one
            #. returns observation by known model

        :param action: action taken by agent
        :return: generated observation

        """

        # (learned) prediction part)
        next_pos = self.learned_model.sample(
            agent_position(self.domain_state), action, num=1
        )

        # known-part
        self.domain_state = self.pomdp.functional_step(
            self.domain_state, GVerseAction(action)
        )[0]

        # merge
        self.domain_state.agent.position = Position(*next_pos)

        return self._obs_rep(self.domain_state)

    def reward(self, action: int, next_state: AugmentedGodState) -> float:
        assert isinstance(next_state, GridversePositionAugmentedState)
        return self.pomdp.reward_function(
            self.domain_state, GVerseAction(action), next_state.domain_state
        )

    def terminal(self, action: int, next_state: AugmentedGodState) -> bool:
        assert isinstance(next_state, GridversePositionAugmentedState)
        return self.pomdp.termination_function(
            self.domain_state, GVerseAction(action), next_state.domain_state
        )
