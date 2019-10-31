""" POMDP dynamics as neural networks """

from collections import deque, namedtuple
from enum import Enum, auto
from itertools import chain
from typing import Tuple, Deque, List

import numpy as np
import torch
import torch.distributions.utils

from po_nrl.agents.neural_networks import Net
from po_nrl.agents.neural_networks.misc import perturb
from po_nrl.environments import ActionSpace
from po_nrl.misc import DiscreteSpace
from po_nrl.pytorch_api import log_tensorboard, device, tensorboard_logging


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
            dropout_rate: float):
        """ Creates a dynamic model

        Args:
             state_space: (`po_nrl.misc.DiscreteSpace`):
             action_space: (`po_nrl.environments.ActionSpace`):
             obs_space: (`po_nrl.misc.DiscreteSpace`):
             network_size: (`int`): number of nodes in hidden layers
             learning_rate: (`float`): learning rate of the optimizer
             batch_size: (`int`): number of interactions to **remember** and update with
             dropout_rate: (`float`): dropout rate of the layers

        """

        self.state_space = state_space
        self.action_space = action_space
        self.obs_space = obs_space

        self.experiences: Deque[Interaction] = deque([], batch_size)

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

        self.optimizer = torch.optim.Adam(
            chain(self.net_t.parameters(), self.net_o.parameters()),
            lr=learning_rate
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.num_batches = 0
        self.learning_rate = learning_rate  # hard to extract from self.optimizer

    def train(self) -> None:
        """ sets to training mode

        * disables dropout layers

        Args:

        RETURNS (`None`):

        """
        self.net_t.train()
        self.net_o.train()

    def eval(self) -> None:
        """ sets to evaluation mode

        * enables dropout layers

        Args:

        RETURNS (`None`):

        """
        self.net_t.eval()
        self.net_o.eval()

    def set_learning_rate(self, learning_rate: float) -> None:
        """ (re)sets the optimizer's learning rate

        Will re-create the optimizer, thus losing its current state

        Args:
             learning_rate: (`float`):

        RETURNS (`None`):

        """
        assert 0 < learning_rate < 1, f'learning rate must be [0,1], not {learning_rate}'

        self.optimizer = torch.optim.Adam(
            chain(self.net_t.parameters(), self.net_o.parameters()),
            lr=learning_rate
        )
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
            conf: FreezeModelSetting = FreezeModelSetting.FREEZE_NONE) -> None:
        """ performs a batch update (single gradient descent step)

        Args:
             states: (`np.ndarray`): (batch_size, state_shape) array of states
             actions: (`np.ndarray`): (batch_size,) array of actions
             next_states: (`np.ndarray`): (batch_size, state_shape) array of (next) states
             obs: (`np.ndarray`): (batch_size, obs_shape) array of observations
             conf: (`FreezeModelSetting`): configurations for training

        """

        self.train()

        actions = [self.action_space.one_hot(a) for a in actions]
        next_states = torch.from_numpy(next_states).to(device())
        obs = torch.from_numpy(obs).to(device())

        # transition model
        if conf == DynamicsModel.FreezeModelSetting.FREEZE_T:
            state_loss = torch.zeros(1)

        else:
            state_action_pairs = torch.from_numpy(np.concatenate(
                [states, actions], axis=1
            )).to(device()).float()
            next_state_logits = self.net_t(state_action_pairs)

            state_loss = torch.stack([
                self.criterion(
                    next_state_logits[:, self.state_space.dim_cumsum[i]:self.state_space.dim_cumsum[i + 1]],
                    next_states[:, i]
                )
                for i in range(self.state_space.ndim)]).sum()

        if conf == DynamicsModel.FreezeModelSetting.FREEZE_O:
            observation_loss = torch.zeros(1)
        else:
            # observation model
            state_action_state_triplets = torch.from_numpy(np.concatenate(
                [states, actions, next_states], axis=1
            )).to(device()).float()
            observation_logits = self.net_o(state_action_state_triplets)

            observation_loss = torch.stack([
                self.criterion(
                    observation_logits[:, self.obs_space.dim_cumsum[i]:self.obs_space.dim_cumsum[i + 1]],
                    obs[:, i])
                for i in range(self.obs_space.ndim)]).sum()

        self.optimizer.zero_grad()
        (state_loss + observation_loss).backward()
        self.optimizer.step()

        if tensorboard_logging():
            log_tensorboard(f'observation_loss/{self}', observation_loss.item(), self.num_batches)
            log_tensorboard(f'transition_loss/{self}', state_loss.item(), self.num_batches)

        self.num_batches += 1

        self.eval()

    def self_learn(self, conf: FreezeModelSetting = FreezeModelSetting.FREEZE_NONE) -> None:
        """ performs a batch update on stored data

        Args:
             conf: (`FreezeModelSetting`): configurations for training

        RETURNS (`None`):

        """
        assert self.experiences, f'cannot self learn without data'

        self.batch_update(*map(np.array, zip(*self.experiences)), conf=conf)  # type: ignore

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

        # sample value for each dimension iteratively
        return np.array([
            torch.multinomial(torch.from_numpy(probs), num).item()
            for probs in self.transition_model(state, action)
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

        # sample value for each dimension iteratively
        return np.array([
            torch.multinomial(torch.from_numpy(probs), num).item()
            for probs in self.observation_model(state, action, next_state)
        ], dtype=int)

    def transition_model(
            self,
            state: np.ndarray,
            action: int) -> List[np.ndarray]:
        """ Returns the transition model (next state) for state-action pair

        element i of the returned list is the transition probability of next
        state dimension i

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

        element i of the returned list is the probability of observation
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

        self.optimizer = torch.optim.Adam(
            chain(self.net_t.parameters(), self.net_o.parameters()),
            lr=self.learning_rate
        )

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
