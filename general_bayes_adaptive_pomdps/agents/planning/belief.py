""" contains algorithms and classes for maintaining beliefs """

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from general_bayes_adaptive_pomdps.agents.neural_networks.neural_pomdps import (
    DynamicsModel,
)
from general_bayes_adaptive_pomdps.agents.planning.particle_filters import (
    Belief,
    create_belief_update,
)
from general_bayes_adaptive_pomdps.environments import Simulator
from general_bayes_adaptive_pomdps.misc import POBNRLogger
from general_bayes_adaptive_pomdps.pytorch_api import (
    log_tensorboard,
    tensorboard_logging,
)
from typing_extensions import Protocol  # pylint: disable=wrong-import-order


class BeliefUpdate(Protocol):
    """ Defines the signature of a update function for a particle """

    def __call__(
        self, belief: Belief, action: np.ndarray, observation: np.ndarray
    ) -> Belief:
        """function call signature for particle update functions

        Args:
             particle_filter: (`Belief`):
             action: (`np.ndarray`):
             observation: (`np.ndarray`):

        RETURNS (`Belief`):

        """


class BeliefAnalysis(Protocol):
    """ defines the protocol for analyzing a belief """

    def __call__(
        self, belief: Belief
    ) -> List[
        Tuple[
            str,
            Union[float, np.ndarray],  # pylint: disable=unsubscriptable-object
        ]
    ]:
        """anylsis `belief` and return some tagged value

        Returns a tuple with 2 values. 1 is a tag/description term of the
        analysis, whereas the 2nd is the values as a result of the analysis

        Args:
             belief: (`Belief`):

        RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

        """


def noop_analysis(
    belief: Belief,
) -> List[
    Tuple[str, Union[float, np.ndarray]]  # pylint: disable=unsubscriptable-object
]:  # pylint: disable=unused-argument
    """default, no-op analysis

    returns a simple non-like thing

    Args:
         belief: (`Belief`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """
    return [("", 0)]


class BeliefManager(POBNRLogger):
    """ manages a belief """

    def __init__(
        self,
        reset_f: Callable[[], Belief],
        update_belief_f: BeliefUpdate,
        episode_reset_f: Optional[  # pylint: disable=unsubscriptable-object
            Callable[[Belief], Belief]
        ] = None,
        belief_analyzer: BeliefAnalysis = noop_analysis,
    ):
        """Maintians a belief

        Manages belief by initializing, updating, and returning it.

        Args:
             reset_f: (`Callable[[], Belief]`): the function to call to reset the belief
             update_belief_f: (`BeliefUpdate`): the function to call to update the belief
             episode_reset_f: (`Optional[Callable[[` `Belief` `], `Belief` ]]`): the episode reset function to call

        Default value for `episode_reset_f` is to do the same as `reset_f`

        """

        POBNRLogger.__init__(self)
        self.episode = 0

        self._reset = reset_f
        self._update = update_belief_f
        self._analyse = belief_analyzer

        if episode_reset_f:
            self._episode_reset = episode_reset_f
        else:
            self._episode_reset = lambda _: self._reset()

        self._belief = self._reset()

    def reset(self) -> None:
        """ resets by sampling new belief """

        self._belief = self._reset()

        if self.log_is_on(POBNRLogger.LogLevel.V3):
            self.log(POBNRLogger.LogLevel.V3, f"Belief reset to {self._belief}")

        self.episode = 0

    def episode_reset(self) -> None:
        """ resets the belief for a new episode """

        self._belief = self._episode_reset(self.particle_filter)

        if self.log_is_on(POBNRLogger.LogLevel.V3):
            self.log(
                POBNRLogger.LogLevel.V3,
                f"Belief reset for new episode {self._belief}",
            )

        if tensorboard_logging():
            for tag, val in self._analyse(self._belief):
                log_tensorboard(tag, val, self.episode)

        self.episode += 1

    def update(self, action: int, observation: np.ndarray):
        """updates belief given action and observation

        Args:
             action: (`int`):
             observation: (`np.ndarray`):

        """

        self._belief = self._update(
            belief=self._belief, action=action, observation=observation
        )

        if self.log_is_on(POBNRLogger.LogLevel.V3):
            self.log(
                POBNRLogger.LogLevel.V3,
                f"BELIEF: update after a({action}), o({observation}): {self._belief}",
            )

    @property
    def particle_filter(self) -> Belief:
        """returns the belief

        RETURNS (`Belief`):

        """
        return self._belief


class ModelUpdate(Protocol):
    """Defines the signature of a model update function

    These functions are used to **augment** the importance sampling update.
    I.e. this function will enable the model to be updated

    """

    def __call__(
        self,
        model: DynamicsModel,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        observation: np.ndarray,
    ) -> None:
        pass


def perturb_parameters(
    model: DynamicsModel,
    state: np.ndarray,  # pylint: disable=unused-argument
    action: np.ndarray,  # pylint: disable=unused-argument
    next_state: np.ndarray,  # pylint: disable=unused-argument
    observation: np.ndarray,  # pylint: disable=unused-argument
    stdev: float,
    freeze_model_setting: DynamicsModel.FreezeModelSetting,
) -> None:
    """A type of belief update: applies gaussian noise to model parameters

    Args:
         model: (`DynamicsModel`):
         _: (`np.ndarray`): ignored
         __: (`np.ndarray`): ignored
         ___: (`np.ndarray`): ignored
         _____: (`np.ndarray`): ignored
         stdev: (`float`): the standard deviation of the applied noise
         freeze_model_setting: (`FreezeModelSetting`)

    RETURNS (`None`):

    """

    model.perturb_parameters(stdev, freeze_model_setting)


def replay_buffer_update(
    model: DynamicsModel,
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    observation: np.ndarray,
    freeze_model_setting: DynamicsModel.FreezeModelSetting,
) -> None:
    """will add transition to model and then invoke a `self learn` step

    Args:
         model: (`DynamicsModel`):
         state: (`np.ndarray`):
         action: (`np.ndarray`):
         next_state: (`np.ndarray`):
         observation: (`np.ndarray`):
         freeze_model_setting: (`FreezeModelSetting`)

    RETURNS (`None`):

    """

    model.add_transition(state, action, next_state, observation)
    model.self_learn(freeze_model_setting)


def backprop_update(
    model: DynamicsModel,
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    observation: np.ndarray,
    freeze_model_setting: DynamicsModel.FreezeModelSetting,
) -> None:
    """A type of belief update: applies a simple backprop call to the model

    Args:
         model: (`DynamicsModel`):
         state: (`np.ndarray`):
         action: (`np.ndarray`):
         next_state: (`np.ndarray`):
         observation: (`np.ndarray`):
         freeze_model_setting: (`FreezeModelSetting`)

    RETURNS (`None`):

    """

    log_loss = False  # online we do not log the loss
    # `batch_update` expects batch_size x ... size. [None] adds a dimension
    model.batch_update(
        state[None],
        action[None],
        next_state[None],
        observation[None],
        log_loss,
        freeze_model_setting,
    )


class ModelUpdatesChain:
    """chains `ModelUpdate` into a single call

    Returns a class that will apply all updates sequentially
    """

    def __init__(self, chain: List[ModelUpdate]):
        """registers which model updates to apply

        Args:
             chain: (`List[ModelUpdate]`):

        """

        self.model_updates = chain

    def __call__(
        self,
        model: DynamicsModel,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        observation: np.ndarray,
    ) -> None:
        """applies all model updates on model, given transition

        Args:
             model: (`DynamicsModel`):
             state: (`np.ndarray`):
             action: (`np.ndarray`):
             next_state: (`np.ndarray`):
             observation: (`np.ndarray`):

        RETURNS (`None`):

        """

        for update in self.model_updates:
            update(model, state, action, next_state, observation)


def create_beliefupdate_for_learning(conf, sim: Simulator) -> BeliefUpdate:
    """returns an importance sampling method depending on the configurations

    Args:
         conf: (`namespace`) program configurations
         sim: (`Simulator`):

    RETURNS (`BeliefUpdate`):

    """

    assert conf.belief in [
        "rejection_sampling",
        "importance_sampling",
    ], f"belief {conf.belief} not legal"

    return create_belief_update(conf, sim)


def get_model_freeze_setting(
    freeze_model: str,
) -> DynamicsModel.FreezeModelSetting:
    """returns the model freeze setting given configuration string

    Args:
         freeze_model: (`str`):

    RETURNS ( `general_bayes_adaptive_pomdps.agents.neural_networks.neural_pomdps.DynamicsModel.FreezeModelSetting`):

    """
    if not freeze_model:
        return DynamicsModel.FreezeModelSetting.FREEZE_NONE

    if freeze_model == "T":
        return DynamicsModel.FreezeModelSetting.FREEZE_T

    if freeze_model == "O":
        return DynamicsModel.FreezeModelSetting.FREEZE_O

    raise ValueError("Wrong value given to freeze model argument")
