"""Runs an example usage of the package"""

import random
from functools import partial
from typing import List

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.baddr.model import (
    BADDr,
    BADDrState,
    backprop_update,
    create_dynamics_model,
    sample_transitions_uniform,
    train_from_samples,
)
from general_bayes_adaptive_pomdps.baddr.neural_networks.neural_pomdps import (
    DynamicsModel,
)
from general_bayes_adaptive_pomdps.domains.tiger import Tiger, TigerPrior


def test_run():
    """Runs a few episodes in BADDr, maintaining a belief through rejection sampling"""

    one_hot_encoding = True
    domain = Tiger(one_hot_encoding)

    num_runs = 4

    # number of particles to approximate belief with
    num_particles = 4

    # (pre-) training parameters
    num_nets = 3
    learning_rate = 0.1
    network_size = batch_size = 32
    dropout_rate = 0.1
    pretrain_epochs = 128

    # pre-train models on prior
    prior = TigerPrior(10, 0, one_hot_encoding)

    models = [
        create_dynamics_model(
            domain.state_space,
            domain.action_space,
            domain.observation_space,
            "SGD",
            learning_rate,
            network_size,
            batch_size,
            dropout_rate,
        )
        for _ in range(num_nets)
    ]

    data_sampler = partial(
        sample_transitions_uniform,
        domain.state_space,
        domain.action_space,
        prior.sample().simulation_step,
    )

    for m in models:
        train_from_samples(m, data_sampler, pretrain_epochs, batch_size)

    # create GBA-POMDP (BADDr) from trained models
    model_updates = [
        partial(
            backprop_update,
            freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_NONE,
        )
    ]
    gba_pomdp = BADDr(
        domain.action_space,
        domain.observation_space,
        domain.sample_start_state,
        domain.reward,
        domain.terminal,
        models,
        model_updates,
    )

    belief = [gba_pomdp.sample_start_state() for _ in range(num_particles)]
    # typical RL loop
    rewards = []

    for _ in range(num_runs):
        while True:

            a = domain.action_space.sample_as_int()
            obs, reward, t = domain.step(a)
            belief = rejection_sampking(gba_pomdp, belief, a, obs)

            # track rewards
            rewards.append(reward)
            if t:
                break

        assert sum(rewards) != 0


def rejection_sampking(
    gba_pomdp: BADDr, b: List[BADDrState], a: int, o: np.ndarray
) -> List[BADDrState]:
    """Implements plain particle filtering rejection sampling"""
    next_b = []

    while len(next_b) < len(b):
        s = random.choice(b)
        next_s, sample_o = gba_pomdp.simulation_step(s, a)

        if (sample_o == o).all():
            next_b.append(next_s)

    return next_b


def some_planner(gba_pomdp: BADDr, b: List[BADDrState]):
    """Implements random planner, but could be any online planner"""
    return gba_pomdp.action_space.sample_as_int()


if __name__ == "__main__":
    pytest.main([__file__])
