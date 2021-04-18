"""tests :mod:`general_bayes_adaptive_pomdps.baddr.neural_networks.neural_pomdps`"""

import copy

import numpy as np
import pytest
import torch

from general_bayes_adaptive_pomdps.baddr.neural_networks.neural_pomdps import (
    DynamicsModel,
    adam_builder,
    get_optimizer_builder,
    sgd_builder,
)
from general_bayes_adaptive_pomdps.core import ActionSpace
from general_bayes_adaptive_pomdps.misc import DiscreteSpace


def setup_dynamics_model():
    s_space = DiscreteSpace([2])
    a_space = ActionSpace(2)
    o_space = DiscreteSpace([2])

    t_net = DynamicsModel.TNet(
        s_space,
        a_space,
        sgd_builder,
        learning_rate=0.1,
        network_size=5,
        dropout_rate=0.5,
    )
    o_net = DynamicsModel.ONet(
        s_space,
        a_space,
        o_space,
        sgd_builder,
        learning_rate=0.1,
        network_size=5,
        dropout_rate=0.5,
    )

    return DynamicsModel(
        state_space=s_space,
        action_space=a_space,
        batch_size=5,
        t_model=t_net,
        o_model=o_net,
    )


def is_equal_models(model_a, model_b) -> bool:
    """returns whether provided models are equal

    Args:
         model_a:
         model_b:
         is_equal: (`bool`)

    RETURNS (`bool`):

    """

    for tensor_a, tensor_b in zip(model_a.parameters(), model_b.parameters()):
        if not torch.equal(tensor_a.data, tensor_b.data):
            return False

        return True


def test_freeze() -> None:
    """tests whether freezing models works properly

    If freeze_O then the observation should not change with update (T should)
    If freeze_T then the observation should not change with update (O should)
    """

    test_model = setup_dynamics_model()

    copied_model = copy.deepcopy(test_model)
    test_model.perturb_parameters(
        freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_O
    )

    assert not is_equal_models(test_model.t.net, copied_model.t.net)  # type: ignore
    assert is_equal_models(test_model.o.net, copied_model.o.net)  # type: ignore

    copied_model = copy.deepcopy(test_model)
    test_model.perturb_parameters(
        freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_T
    )

    assert is_equal_models(test_model.t.net, copied_model.t.net)  # type: ignore
    assert not is_equal_models(test_model.o.net, copied_model.o.net)  # type: ignore

    copied_model = copy.deepcopy(test_model)
    test_model.batch_update(
        np.array([[0.5]]),
        np.array([0]),
        np.array([[0]]),
        np.array([[0]]),
        conf=DynamicsModel.FreezeModelSetting.FREEZE_O,
    )

    assert not is_equal_models(test_model.t.net, copied_model.t.net)  # type: ignore
    assert is_equal_models(test_model.o.net, copied_model.o.net)  # type: ignore

    copied_model = copy.deepcopy(test_model)

    assert is_equal_models(test_model.t.net, copied_model.t.net)  # type: ignore
    test_model.batch_update(
        np.array([[0]]),
        np.array([0]),
        np.array([[0]]),
        np.array([[0]]),
        conf=DynamicsModel.FreezeModelSetting.FREEZE_T,
    )

    assert not is_equal_models(test_model.o.net, copied_model.o.net)  # type: ignore
    assert is_equal_models(test_model.t.net, copied_model.t.net)  # type: ignore


def test_copy() -> None:
    """tests the copy function of the `general_bayes_adaptive_pomdps.baddr.neural_networks.neural_pomdps.DynamicsModel`

    Basically double checking whether the standard implementation works as
    **I** expect

    Args:

    RETURNS (`None`):

    """
    test_model = setup_dynamics_model()

    copied_model = copy.deepcopy(test_model)

    assert is_equal_models(
        test_model.t.net, copied_model.t.net  # type:ignore
    )
    assert is_equal_models(
        test_model.o.net, copied_model.o.net  # type:ignore
    )

    test_model.perturb_parameters()

    assert not is_equal_models(test_model.t.net, copied_model.t.net)  # type: ignore
    assert not is_equal_models(test_model.o.net, copied_model.o.net)  # type: ignore

    copied_model = copy.deepcopy(test_model)

    assert is_equal_models(test_model.t.net, copied_model.t.net)  # type: ignore
    assert is_equal_models(test_model.o.net, copied_model.o.net)  # type: ignore

    copied_model.batch_update(
        np.array([[1]]), np.array([0]), np.array([[0]]), np.array([[1]])
    )

    assert not is_equal_models(test_model.t.net, copied_model.t.net)  # type: ignore
    assert not is_equal_models(test_model.o.net, copied_model.o.net)  # type: ignore


def test_optimizer_builder() -> None:
    """ Simple tests to ensure the correct builder is returned """
    assert get_optimizer_builder("SGD") == sgd_builder
    assert get_optimizer_builder("Adam") == adam_builder
    with pytest.raises(ValueError):
        get_optimizer_builder("a wrong value")


def test_save_and_load_dynamics_model(tmp_path):
    fname = tmp_path / ".tar"

    model_a = setup_dynamics_model()
    model_a.save(fname)

    model_b = setup_dynamics_model()
    assert not is_equal_models(model_a.t.net, model_b.t.net) and not is_equal_models(
        model_a.o.net, model_b.o.net
    )

    model_b.load(fname)
    assert is_equal_models(model_a.t.net, model_b.t.net) and is_equal_models(
        model_a.o.net, model_b.o.net
    )


if __name__ == "__main__":
    pytest.main([__file__])
