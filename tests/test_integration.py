"""Runs an example usage of the package"""

import random
from functools import partial
from typing import List

import numpy as np
import pytest

from general_bayes_adaptive_pomdps.core import GeneralBAPOMDP
from general_bayes_adaptive_pomdps.domains.tiger import (
    Tiger,
    TigerPrior,
    create_tabular_prior_counts,
)
from general_bayes_adaptive_pomdps.models.baddr import (
    BADDr,
    BADDrState,
    backprop_update,
    create_dynamics_model,
    sample_transitions_uniform,
    train_from_samples,
)
from general_bayes_adaptive_pomdps.models.neural_networks.neural_pomdps import (
    DynamicsModel,
)
from general_bayes_adaptive_pomdps.models.tabular_bapomdp import TabularBAPOMDP


def test_tabular_bapomdp():
    """Runs a few episodes of the tabular BA-POMDP using random actions with rejection sampling"""

    domain = Tiger(one_hot_encode_observation=False)

    tbapomdp = TabularBAPOMDP(
        domain.state_space,
        domain.action_space,
        domain.observation_space,
        domain.sample_start_state,
        domain.reward,
        domain.terminal,
        create_tabular_prior_counts(),
    )

    run_gba_pomdp(tbapomdp, domain)


def test_baddr():
    """Runs a few episodes in BADDr, maintaining a belief through rejection sampling"""

    one_hot_encoding = True
    domain = Tiger(one_hot_encoding)

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
    baddr = BADDr(
        domain.action_space,
        domain.observation_space,
        domain.sample_start_state,
        domain.reward,
        domain.terminal,
        models,
        model_updates,
    )

    run_gba_pomdp(baddr, domain)


def run_gba_pomdp(gba_pomdp: GeneralBAPOMDP, domain: Tiger):

    num_particles = 8  # number of particles to approximate belief with
    num_runs = 3

    belief = [gba_pomdp.sample_start_state() for _ in range(num_particles)]

    # typical RL loop
    for _ in range(num_runs):
        rewards = []
        for i in range(4):

            a = Tiger.LISTEN if i < 3 else random.choice([0, 1])
            obs, reward, t = domain.step(a)

            print(f"a({a} => o({obs}), r({reward}), terminal({t})")

            belief = rejection_sampling(gba_pomdp, belief, a, obs)

            # track rewards
            rewards.append(reward)
            if t:
                break

        assert sum(rewards) != 0
        assert len(rewards) == 4


def rejection_sampling(
    gba_pomdp: GeneralBAPOMDP, b: List[BADDrState], a: int, o: np.ndarray
) -> List[BADDrState]:
    """Implements plain particle filtering rejection sampling"""
    next_b = []

    while len(next_b) < len(b):
        s = random.choice(b)
        next_s, sample_o = gba_pomdp.simulation_step(s, a)

        if (sample_o == o).all():
            next_b.append(next_s)

    return next_b


if __name__ == "__main__":
    pytest.main([__file__])
