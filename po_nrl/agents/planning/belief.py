""" contains algorithms and classes for maintaining beliefs """

from typing import Callable, Optional, Union, Tuple, List
from functools import partial

from typing_extensions import Protocol
import numpy as np

from po_nrl.misc import POBNRLogger
from po_nrl.environments import Simulator
from po_nrl.pytorch_api import log_tensorboard, tensorboard_logging

from po_nrl.agents.neural_networks.neural_pomdps import DynamicsModel
from po_nrl.agents.planning.particle_filters import ParticleFilter, WeightedFilter, WeightedParticle
from po_nrl.domains.learned_environments import NeuralEnsemblePOMDP


class BeliefUpdate(Protocol):
    """ Defines the signature of a update function for a particle """

    def __call__(
            self,
            belief: ParticleFilter,
            action: np.ndarray,
            observation: np.ndarray) -> ParticleFilter:
        """ function call signature for particle update functions

        Args:
             particle_filter: (`ParticleFilter`):
             action: (`np.ndarray`):
             observation: (`np.ndarray`):

        RETURNS (`ParticleFilter`):

        """


class BeliefAnalysis(Protocol):
    """ defines the protocol for analyzing a belief """

    def __call__(self, belief: ParticleFilter) -> List[Tuple[str, Union[float, np.ndarray]]]:
        """ anylsis `belief` and return some tagged value

        Returns a tuple with 2 values. 1 is a tag/description term of the
        analysis, whereas the 2nd is the values as a result of the analysis

        Args:
             belief: (`ParticleFilter`):

        RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

        """


def noop_analysis(
        belief: ParticleFilter) -> List[Tuple[str, Union[float, np.ndarray]]]:  # pylint: disable=unused-argument
    """  default, no-op analysis

    returns a simple non-like thing

    Args:
         belief: (`ParticleFilter`):

    RETURNS (`List[Tuple[str, Union[float,np.ndarray]]]`):

    """
    return [('', 0)]


class BeliefManager(POBNRLogger):
    """ manages a belief """

    def __init__(
            self,
            reset_f: Callable[[], ParticleFilter],
            update_belief_f: BeliefUpdate,
            episode_reset_f: Optional[Callable[[ParticleFilter], ParticleFilter]] = None,
            belief_analyzer: BeliefAnalysis = noop_analysis):
        """ Maintians a belief

        Manages belief by initializing, updating, and returning it.

        Args:
             reset_f: (`Callable[[], ParticleFilter]`): the function to call to reset the belief
             update_belief_f: (`BeliefUpdate`): the function to call to update the belief
             episode_reset_f: (`Optional[Callable[[` `ParticleFilter` `], `ParticleFilter` ]]`): the episode reset function to call

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
            self.log(POBNRLogger.LogLevel.V3, f"Belief reset for new episode {self._belief}")

        if tensorboard_logging():
            for tag, val in self._analyse(self._belief):
                log_tensorboard(tag, val, self.episode)

        self.episode += 1

    def update(self, action: int, observation: np.ndarray):
        """ updates belief given action and observation

        Args:
             action: (`int`):
             observation: (`np.ndarray`):

        """

        self._belief = self._update(belief=self._belief, action=action, observation=observation)

        if self.log_is_on(POBNRLogger.LogLevel.V3):
            self.log(POBNRLogger.LogLevel.V3, f"BELIEF: update after a({action}), o({observation}): {self._belief}")

    @property
    def particle_filter(self) -> ParticleFilter:
        """ returns the belief

        RETURNS (`ParticleFilter`):

        """
        return self._belief


def rejection_sampling(
        belief: ParticleFilter,
        action: np.ndarray,
        observation: np.ndarray,
        sim: Simulator) -> ParticleFilter:
    """ performs vanilla rejection sampling as belief update

    Args:
         belief: (`ParticleFilter`):
         action: (`np.ndarray`):
         observation: (`np.ndarray`):
         sim: (`Simulator`):

    RETURNS (`ParticleFilter`):

    """

    next_belief = type(belief)()

    while next_belief.size < belief.size:

        state = belief.sample()
        transition = sim.simulation_step(state, action)

        if np.all(transition.observation == observation):
            next_belief.add_particle(transition.state)

    return next_belief


def importance_sampling(
        belief: ParticleFilter,
        action: np.ndarray,
        observation: np.ndarray) -> WeightedFilter:
    """ applies importance sampling **with resampling** as belief update

    Assumes belief is over state and models, i.e.
    `pobnrl.domains.learned_environments.NeuralEnsemblePOMDP.AugmentedState`

    Args:
         belief: (`ParticleFilter`):
         action: (`np.ndarray`):
         observation: (`np.ndarray`):

    RETURNS (`WeightedFilter`):

    """

    next_belief = WeightedFilter()

    for _ in range(belief.size):

        # This step functions as the resampling step
        state = belief.sample()

        next_domain_state = state.model.sample_state(state.domain_state, action)

        weight = state.model.observation_prob(
            state.domain_state, action, next_domain_state, observation
        )

        next_belief.add_weighted_particle(WeightedParticle(
            NeuralEnsemblePOMDP.AugmentedState(next_domain_state, state.model),
            weight
        ))

    return next_belief


class ModelUpdate(Protocol):
    """ Defines the signature of a model update function

    These functions are used to **augment** the importance sampling update.
    I.e. this function will enable the model to be updated

    """

    def __call__(
            self,
            model: DynamicsModel,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            observation: np.ndarray) -> None:
        pass


def perturb_parameters(
        model: DynamicsModel,
        state: np.ndarray,  # pylint: disable=unused-argument
        action: np.ndarray,  # pylint: disable=unused-argument
        next_state: np.ndarray,  # pylint: disable=unused-argument
        observation: np.ndarray,  # pylint: disable=unused-argument
        stdev: float) -> None:
    """ A type of belief update: applies gaussian noise to model parameters

    Args:
         model: (`DynamicsModel`):
         _: (`np.ndarray`): ignored
         __: (`np.ndarray`): ignored
         ___: (`np.ndarray`): ignored
         _____: (`np.ndarray`): ignored
         stdev: (`float`): the standard deviation of the applied noise

    RETURNS (`None`):

    """

    model.perturb_parameters(stdev)


def backprop_update(
        model: DynamicsModel,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        observation: np.ndarray) -> None:
    """ A type of belief update: applies a simple backprop call to the model

    Args:
         model: (`DynamicsModel`):
         state: (`np.ndarray`):
         action: (`np.ndarray`):
         next_state: (`np.ndarray`):
         observation: (`np.ndarray`):

    RETURNS (`None`):

    """

    # `batch_update` expects batch_size x ... size. [None] adds a dimension
    model.batch_update(state[None], action[None], next_state[None], observation[None])


class ModelUpdatesChain():
    """ chains `ModelUpdate` into a single call

    Returns a class that will apply all updates sequentially
    """

    def __init__(self, chain: List[ModelUpdate]):
        """ registers which model updates to apply

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
            observation: np.ndarray) -> None:
        """ applies all model updates on model, given transition

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


def augmented_rejection_sampling(
        belief: ParticleFilter,
        action: np.ndarray,
        observation: np.ndarray,
        update_model: ModelUpdate) -> ParticleFilter:
    """ rejection sampling augmented with additional update methods

    Args:
         belief: (`ParticleFilter`): current belief
         action: (`np.ndarray`):  taken action
         observation: (`np.ndarray`): perceived observation
         update_model: (`ModelUpdate`): the update to apply

    RETURNS (`ParticleFilter`):

    """

    next_belief = type(belief)()

    while next_belief.size < belief.size:

        state = belief.sample()
        sample_state, sample_observation \
            = state.model.simulation_step(state.domain_state, action)

        if np.all(sample_observation == observation):

            next_model = state.model.copy()
            update_model(
                model=next_model,
                state=state.domain_state,
                action=action,
                next_state=sample_state,
                observation=observation
            )

            next_belief.add_particle(
                NeuralEnsemblePOMDP.AugmentedState(sample_state, next_model)
            )

    return next_belief


def augmented_importance_sampling(
        belief: ParticleFilter,
        action: np.ndarray,
        observation: np.ndarray,
        update_model: ModelUpdate) -> ParticleFilter:
    """ Core algorithm for this project. Updates the model during belief update

    Assuming a belief p(state, dynamics), this function will compute an
    importance sampling belief update. It differs from `importance_sampling` in
    that it also updates the models. The specific update that is applied is
    determined by the `update_model` parameter.

    Args:
         belief: (`ParticleFilter`):
         action: (`np.ndarray`):
         observation: (`np.ndarray`):
         update_model: (`ModelUpdate`):

    RETURNS (`ParticleFilter`):

    """

    next_belief = WeightedFilter()

    for _ in range(belief.size):

        # This step functions as the resampling step
        state = belief.sample()
        next_model = state.model.copy()

        next_domain_state = state.model.sample_state(state.domain_state, action)
        update_model(
            model=next_model,
            state=state.domain_state,
            action=action,
            next_state=next_domain_state,
            observation=observation
        )

        weight = next_model.observation_prob(
            state.domain_state, action, next_domain_state, observation
        )

        next_belief.add_weighted_particle(WeightedParticle(
            NeuralEnsemblePOMDP.AugmentedState(next_domain_state, next_model),
            weight
        ))

    return next_belief


def belief_update_factory(
        belief: str,
        perturb_stdev: float,
        backprop: bool,
        sim: Simulator) -> BeliefUpdate:
    """ returns an importance sampling method depending on the configurations

    Args:
         belief: (`str`):
         perturb_stdev: (`float`): the amount of param perturbation during updates
         backprop: (`bool`): whether to apply backprop during update
         sim: (`Simulator`):

    RETURNS (`BeliefUpdate`):

    """

    assert belief in ["rejection_sampling", "importance_sampling"], \
        f"belief {belief} not legal"

    # basic, no enhancements
    if perturb_stdev == 0 and not backprop:
        if belief == 'importance_sampling':
            return importance_sampling
        if belief == 'rejection_sampling':
            return partial(rejection_sampling, sim=sim)

    # set filter method
    if belief == 'importance_sampling':
        filter_method = augmented_importance_sampling
    elif belief == 'rejection_sampling':
        filter_method = augmented_rejection_sampling

    # set model update method
    updates: List[ModelUpdate] = []
    if backprop:
        updates.append(backprop_update)
    if not perturb_stdev == 0:
        updates.append(partial(perturb_parameters, stdev=perturb_stdev))

    return partial(filter_method, update_model=ModelUpdatesChain(updates))
