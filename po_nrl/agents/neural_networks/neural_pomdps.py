""" POMDP dynamics as neural networks """

from collections import deque, namedtuple
from enum import Enum, auto
from typing import Tuple, Deque, List, Any
from typing_extensions import Protocol

import numpy as np
import torch
import torch.distributions.utils

from po_nrl.agents.neural_networks import Net
from po_nrl.agents.neural_networks.misc import perturb
from po_nrl.environments import ActionSpace
from po_nrl.misc import DiscreteSpace
from po_nrl.pytorch_api import log_tensorboard, device


class Interaction(
        namedtuple('interaction', 'state action next_state observation')):
    """ transition in environment

        Contains:
             state: (`np.ndarray`):
             action: (`int`):
             next_state: (`np.ndarray`):
             observation: (`np.ndarray`):
    """
    __slots__ = ()  # required to keep lightweight implementation of namedtuple


class OptimizerBuilder(Protocol):
    """ Defines the signature of optimizer builder"""

    def __call__(self, parameters: Any, learning_rate: float) -> torch.optim.Optimizer:
        """ function call signature for building an optimizer

        Args:
             parameters: (torch parameters): the parameters to optimize
             learning_rate: (`float`): learning rate of the optimizer

        RETURNS (`torch.optim.Optimizer`):

        """


def sgd_builder(parameters: Any, learning_rate: float) -> torch.optim.Optimizer:
    """ builds the torch SGD optimizer to update `parameters` with `learning_rate` stepsize

        Args:
             parameters: (torch parameters): the parameters to optimize
             learning_rate: (`float`): learning rate of the optimizer

        RETURNS (`torch.optim.Optimizer`): Torch SGD optimizer

        """
    return torch.optim.SGD(parameters, lr=learning_rate)


def adam_builder(parameters: Any, learning_rate: float) -> torch.optim.Optimizer:
    """ builds the torch Adam optimizer to update `parameters` with `learning_rate` stepsize

        Args:
             parameters: (torch parameters): the parameters to optimize
             learning_rate: (`float`): learning rate of the optimizer

        RETURNS (`torch.optim.Optimizer`): Torch Adam optimizer

        """
    return torch.optim.Adam(parameters, lr=learning_rate)


def get_optimizer_builder(option: str) -> OptimizerBuilder:
    """ Returns the appropriate optimizer builder

        Args:
             state_space: (`po_nrl.misc.DiscreteSpace`):

        RETURNS (`po_nrl.agents.neural_networks.OptimizerBuilder`):

    """
    if option == 'SGD':
        return sgd_builder
    if option == 'Adam':
        return adam_builder

    raise ValueError(f'Undefined optimizer {option}')


class DynamicsModel:
    """ A neural network representing POMDP dynamics (s,a) -> p(s',o) """

    class FreezeModelSetting(Enum):
        """ setting for training """
        FREEZE_NONE = auto()
        FREEZE_T = auto()
        FREEZE_O = auto()

    def __init__(
            self,
            state_space: DiscreteSpace,
            action_space: ActionSpace,
            obs_space: DiscreteSpace,
            network_size: int,
            learning_rate: float,
            batch_size: int,
            dropout_rate: float,
            optimizer_builder: OptimizerBuilder):
        """ Creates a dynamic model

        Args:
             state_space: (`po_nrl.misc.DiscreteSpace`):
             action_space: (`po_nrl.environments.ActionSpace`):
             obs_space: (`po_nrl.misc.DiscreteSpace`):
             network_size: (`int`): number of nodes in hidden layers
             learning_rate: (`float`): learning rate of the optimizers
             batch_size: (`int`): number of interactions to **remember** and update with
             dropout_rate: (`float`): dropout rate of the layers
             optimizer_builder: (`OptimizerBuilder`): builder function for optimizer

        """

        self.state_space = state_space
        self.action_space = action_space
        self.obs_space = obs_space

        self.experiences: Deque[Interaction] = deque([], batch_size)

        self.num_batches = 0
        self.learning_rate = learning_rate

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_builder = optimizer_builder

        self.net_t = Net(
            input_size=self.state_space.ndim + self.action_space.n,
            output_size=np.sum(self.state_space.size),
            layer_size=network_size,
            dropout_rate=dropout_rate,
        ).to(device())

        self.net_o = Net(
            input_size=self.state_space.ndim * 2 + self.action_space.n,
            output_size=np.sum(self.obs_space.size),
            layer_size=network_size,
            dropout_rate=dropout_rate,
        ).to(device())

        self.t_optimizer = self.optimizer_builder(self.net_t.parameters(), self.learning_rate)
        self.o_optimizer = self.optimizer_builder(self.net_o.parameters(), self.learning_rate)

    def set_learning_rate(self, learning_rate: float) -> None:
        """ (re)sets the optimizers' learning rate

        Will re-create the optimizers, thus losing its current state

        Args:
             learning_rate: (`float`):

        RETURNS (`None`):

        """
        assert 0 < learning_rate < 1, f'learning rate must be [0,1], not {learning_rate}'

        self.t_optimizer = self.optimizer_builder(self.net_t.parameters(), learning_rate)
        self.o_optimizer = self.optimizer_builder(self.net_o.parameters(), learning_rate)

        self.learning_rate = learning_rate

    def simulation_step(self, state: np.array, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """ The simulation step of this dynamics model: S x A -> S, O

        Args:
             state: (`np.array`): input state
             action: (`int`): chosen action

        RETURNS (`Tuple[np.ndarray, np.ndarray]`): [state, observation]

        """

        assert self.state_space.contains(state),\
            f"{state} not in {self.state_space}"
        assert self.action_space.contains(action),\
            f"{action} not in {self.action_space}"

        next_state = self.sample_state(state, action)
        observation = self.sample_observation(state, action, next_state)

        return next_state, observation

    def batch_update(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            next_states: np.ndarray,
            obs: np.ndarray,
            log_loss: bool = False,
            conf: FreezeModelSetting = FreezeModelSetting.FREEZE_NONE) -> None:
        """ performs a batch update (single gradient descent step)

        Args:
             states: (`np.ndarray`): (batch_size, state_shape) array of states
             actions: (`np.ndarray`): (batch_size,) array of actions
             next_states: (`np.ndarray`): (batch_size, state_shape) array of (next) states
             obs: (`np.ndarray`): (batch_size, obs_shape) array of observations
             log_loss: (`bool`): whether to log the loss
             conf: (`FreezeModelSetting`): configurations for training

        """

        actions = [self.action_space.one_hot(a) for a in actions]
        next_states = torch.from_numpy(next_states).to(device())
        obs = torch.from_numpy(obs).to(device())

        # transition model
        if conf != DynamicsModel.FreezeModelSetting.FREEZE_T:

            state_action_pairs = torch.from_numpy(np.concatenate(
                [states, actions], axis=1
            )).to(device()).float()
            next_state_logits = self.net_t(state_action_pairs)

            loss = torch.stack([
                self.criterion(
                    next_state_logits[:, self.state_space.dim_cumsum[i]:self.state_space.dim_cumsum[i + 1]],
                    next_states[:, i]
                )
                for i in range(self.state_space.ndim)]).sum()

            self.t_optimizer.zero_grad()
            loss.backward()
            self.t_optimizer.step()

            if log_loss:
                log_tensorboard(f'transition_loss/{self}', loss.item(), self.num_batches)

        # observation model
        if conf != DynamicsModel.FreezeModelSetting.FREEZE_O:

            state_action_state_triplets = torch.from_numpy(np.concatenate(
                [states, actions, next_states], axis=1
            )).to(device()).float()
            observation_logits = self.net_o(state_action_state_triplets)

            loss = torch.stack([
                self.criterion(
                    observation_logits[:, self.obs_space.dim_cumsum[i]:self.obs_space.dim_cumsum[i + 1]],
                    obs[:, i])
                for i in range(self.obs_space.ndim)]).sum()

            self.o_optimizer.zero_grad()
            loss.backward()
            self.o_optimizer.step()

            if log_loss:
                log_tensorboard(f'observation_loss/{self}', loss.item(), self.num_batches)

        self.num_batches += 1

    def self_learn(self, conf: FreezeModelSetting = FreezeModelSetting.FREEZE_NONE) -> None:
        """ performs a batch update on stored data

        Args:
             conf: (`FreezeModelSetting`): configurations for training

        RETURNS (`None`):

        """
        assert self.experiences, f'cannot self learn without data'

        log_loss = False  # no logging of loss online
        self.batch_update(*map(np.array, zip(*self.experiences)), log_loss, conf)  # type: ignore

    def add_transition(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray,
            observation: np.ndarray) -> None:
        """ stores the given transition

        `this` uses this data to learn

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             next_state: (`np.ndarray`):
             observation: (`np.ndarray`):

        RETURNS (`None`):

        """

        self.experiences.append(
            Interaction(state, action, next_state, observation)
        )

    def sample_state(self, state: np.ndarray, action: int, num: int = 1) -> np.ndarray:
        """ samples next state given current and action

        Args:
             state: (`np.ndarray`): current state
             action: (`int`): taken action
             num: (`int`): number of samples

        RETURNS (`np.ndarray`): next state

        """

        transition_probabilities = self.transition_model(state, action)

        # sample value for each dimension iteratively
        return np.array([
            torch.multinomial(torch.from_numpy(probs), num, replacement=True).item()
            for probs in transition_probabilities
        ], dtype=int)

    def sample_observation(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray,
            num: int = 1) -> np.ndarray:
        """ samples an observation given state - action - next state triple

        Args:
             state: (`np.ndarray`): state at t
             action: (`int`): taken action at t
             next_state: (`np.ndarray`): state t + 1
             num: (`int`): number of samples

        RETURNS (`np.ndarray`): observation at t + 1

        """

        observation_probabilities = self.observation_model(state, action, next_state)

        # sample value for each dimension iteratively
        return np.array([
            torch.multinomial(torch.from_numpy(probs), num, replacement=True).item()
            for probs in observation_probabilities
        ], dtype=int)

    def transition_model(
            self,
            state: np.ndarray,
            action: int) -> List[np.ndarray]:
        """ Returns the transition model (next state) for state-action pair

        Element i of the returned list is the (batch, dim_size) probabilities
        of dimension i as a numpy array

        Args:
             state: (`np.ndarray`):
             action: (`int`):

        RETURNS (`List[np.ndarray]`): [#dim x dim_size] list

        """

        state_action_pair = torch.from_numpy(np.concatenate(
            [state, self.action_space.one_hot(action)]
        )).to(device()).float()

        with torch.no_grad():
            logits = self.net_t(state_action_pair)

            return [
                torch.distributions.utils.logits_to_probs(
                    logits[self.state_space.dim_cumsum[i]:self.state_space.dim_cumsum[i + 1]]
                ).numpy() for i in range(self.state_space.ndim)
            ]

    def observation_model(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray) -> List[np.ndarray]:
        """ Returns the observation model of a transition

        Element i of the returned list is the probability of observation
        dimension i

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             next_state: (`np.ndarray`):

        RETURNS (`List[np.ndarray]`): [#dim x dim_size] list

        """

        state_action_state = torch.from_numpy(np.concatenate(
            [state, self.action_space.one_hot(action), next_state]
        )).to(device()).float()

        with torch.no_grad():
            logits = self.net_o(state_action_state)

            return [
                torch.distributions.utils.logits_to_probs(
                    logits[self.obs_space.dim_cumsum[i]:self.obs_space.dim_cumsum[i + 1]]
                ).numpy() for i in range(self.obs_space.ndim)
            ]

    def reset(self) -> None:
        """ resets the networks """
        self.experiences.clear()
        self.net_t.random_init_parameters()
        self.net_o.random_init_parameters()
        self.num_batches = 0

        self.o_optimizer = self.optimizer_builder(self.net_o.parameters(), self.learning_rate)
        self.t_optimizer = self.optimizer_builder(self.net_t.parameters(), self.learning_rate)

    def perturb_parameters(
            self,
            stdev: float = .1,
            freeze_model_setting: FreezeModelSetting = FreezeModelSetting.FREEZE_NONE) -> None:
        """ perturb parameters of model

        Args:
             stdev: (`float`): the standard deviation of the pertubation
             freeze_model_setting: (`po_nrl.agents.neural_networks.neural_pomdps.DynamicsModel.FreezeModelSetting`)

        RETURNS (`None`):

        """

        with torch.no_grad():
            if freeze_model_setting != DynamicsModel.FreezeModelSetting.FREEZE_T:
                for param in self.net_t.parameters():
                    param.set_(perturb(param, stdev))

            if freeze_model_setting != DynamicsModel.FreezeModelSetting.FREEZE_O:
                for param in self.net_o.parameters():
                    param.set_(perturb(param, stdev))
