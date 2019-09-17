""" POMDP dynamics as neural networks """

from itertools import chain
from typing import Tuple
import copy
import re

import numpy as np
import torch
import torch.distributions.utils

from agents.neural_networks import Net
from environments import ActionSpace
from misc import DiscreteSpace
from pytorch_api import log_tensorboard, device


class DynamicsModel():
    """ A neural network representing POMDP dynamics (s,a) -> p(s',o) """

    def __init__(
            self,
            state_space: DiscreteSpace,
            action_space: ActionSpace,
            obs_space: DiscreteSpace,
            network_size: int,
            learning_rate: float,
            name: str):
        """ Creates a dynamic model

        Args:
             state_space: (`pobnrl.misc.DiscreteSpace`):
             action_space: (`pobnrl.environments.ActionSpace`):
             obs_space: (`pobnrl.misc.DiscreteSpace`):
             network_size: (`int`): number of nodes in hidden layers
             learning_rate: (`float`): learning rate of the optimizer
             name: (`str`): name of the network

        """

        self.name = name

        self.state_space = state_space
        self.action_space = action_space
        self.obs_space = obs_space

        self.net_t = Net(
            input_size=self.state_space.ndim + self.action_space.n,
            output_size=np.sum(self.state_space.size),
            layer_size=network_size
        ).to(device())

        self.net_o = Net(
            input_size=self.state_space.ndim * 2 + self.action_space.n,
            output_size=np.sum(self.obs_space.size),
            layer_size=network_size
        ).to(device())

        self.optimizer = torch.optim.Adam(
            chain(self.net_t.parameters(), self.net_o.parameters()),
            lr=learning_rate
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.num_batches = 0
        self.learning_rate = learning_rate  # hard to extract from self.optimizer

    def simulation_step(self, state: np.array, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """ The simulation step of this dynamics model: S x A -> S, O

        Args:
             state: (`np.array`): input state
             action: (`int`): chosen action

        RETURNS (`Tuple[np.ndarray, np.ndarray]`): [state, observation]

        """

        assert self.state_space.contains(state), f"{state} not in {self.state_space}"
        assert self.action_space.contains(action), f"{action} not in {self.action_space}"

        next_state = self.sample_state(state, action)
        observation = self.sample_observation(state, action, next_state)

        return next_state, observation

    def batch_update(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            next_states: np.ndarray,
            obs: np.ndarray) -> None:
        """ performs a batch update (single gradient descent step)

        Args:
             states: (`np.ndarray`): (batch_size, state_shape) array of states
             actions: (`np.ndarray`): (batch_size,) array of actions
             next_states: (`np.ndarray`): (batch_size, state_shape) array of (next) states
             obs: (`np.ndarray`): (batch_size, obs_shape) array of observations

        """

        actions = [self.action_space.one_hot(a) for a in actions]
        next_states = torch.from_numpy(next_states).to(device())
        obs = torch.from_numpy(obs).to(device())

        state_action_pairs = torch.from_numpy(np.concatenate(
            [states, actions], axis=1
        )).to(device()).float()

        state_action_state_triplets = torch.from_numpy(np.concatenate(
            [states, actions, next_states], axis=1
        )).to(device()).float()

        next_state_logits = self.net_t(state_action_pairs)
        observation_logits = self.net_o(state_action_state_triplets)

        state_loss = torch.stack([
            self.criterion(
                next_state_logits[:, self.state_space.dim_cumsum[i]:self.state_space.dim_cumsum[i + 1]],
                next_states[:, i]
            )
            for i in range(self.state_space.ndim)]).sum()

        observation_loss = torch.stack([
            self.criterion(
                observation_logits[:, self.obs_space.dim_cumsum[i]:self.obs_space.dim_cumsum[i + 1]],
                obs[:, i])
            for i in range(self.obs_space.ndim)]).sum()

        self.optimizer.zero_grad()
        (state_loss + observation_loss).backward()
        self.optimizer.step()

        log_tensorboard(f'observation_loss/{self.name}', observation_loss.item(), self.num_batches)
        log_tensorboard(f'transition_loss/{self.name}', state_loss.item(), self.num_batches)

        self.num_batches += 1

    def sample_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """ samples next state given current and action

        Args:
             state: (`np.ndarray`): current state
             action: (`int`): taken action

        RETURNS (`np.ndarray`): next state

        """

        state_action = torch.from_numpy(np.concatenate(
            [state, self.action_space.one_hot(action)]
        )).to(device()).float()

        # sample value for each dimension iteratively
        with torch.no_grad():
            logits = self.net_t(state_action)

            return np.array([
                torch.multinomial(
                    torch.distributions.utils.logits_to_probs(
                        logits[self.state_space.dim_cumsum[i]:self.state_space.dim_cumsum[i + 1]]
                    ),
                    1).item() for i in range(self.state_space.ndim)
            ], dtype=int)

    def sample_observation(self, state: np.ndarray, action: int, next_state: np.ndarray) -> np.ndarray:
        """ samples an observation given state - action - next state triple

        Args:
             state: (`np.ndarray`): state at t
             action: (`int`): taken action at t
             next_state: (`np.ndarray`): state t + 1

        RETURNS (`np.ndarray`): observation at t + 1

        """

        state_action_state = torch.from_numpy(np.concatenate(
            [state, self.action_space.one_hot(action), next_state]
        )).to(device()).float()

        with torch.no_grad():
            logits = self.net_o(state_action_state)

            # sample value for each dimension iteratively
            return np.array([
                torch.multinomial(
                    torch.distributions.utils.logits_to_probs(
                        logits[self.obs_space.dim_cumsum[i]:self.obs_space.dim_cumsum[i + 1]]
                    ),
                    1).item() for i in range(self.obs_space.ndim)
            ], dtype=int)

    def observation_prob(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray,
            observation: np.ndarray) -> float:
        """ Returns the observation probability of a transition

        Args:
             state: (`np.ndarray`):
             action: (`int`):
             next_state: (`np.ndarray`):
             observation: (`np.ndarray`):

        RETURNS (`float`):

        """

        state_action_state = torch.from_numpy(np.concatenate(
            [state, self.action_space.one_hot(action), next_state]
        )).to(device()).float()

        with torch.no_grad():
            logits = self.net_o(state_action_state)

            return np.prod([
                torch.distributions.utils.logits_to_probs(
                    logits[self.obs_space.dim_cumsum[i]:self.obs_space.dim_cumsum[i + 1]]
                )[observation[i]].item() for i in range(self.obs_space.ndim)
            ])

    def reset(self) -> None:
        """ resets the networks """
        self.net_t.random_init_parameters()
        self.net_o.random_init_parameters()
        self.num_batches = 0

    def copy(self) -> 'DynamicsModel':
        """ copies self

        Takes care of which members to copy deep and afterwards links pytorch
        optimizer with the correct parameters

        Args:

        RETURNS (`DynamicsModel`):

        """

        copied_model = self

        copied_model.net_o = copy.deepcopy(self.net_o)
        copied_model.net_t = copy.deepcopy(self.net_t)

        copied_model.optimizer = torch.optim.Adam(
            chain(copied_model.net_t.parameters(), copied_model.net_o.parameters()),
            self.learning_rate
        )

        # some magic to maintain name consistency:
        # create (name)-copy-1 or increment counter if already available

        name_splits = re.split(r'(-copy-\d+)', self.name)
        if len(name_splits) == 1:
            copied_model.name = f'{name_splits[0]}-copy-1'
        else:
            assert len(name_splits) == 3, f'not sure how we got to network name {self.name}'

            i = int(re.split(r'(\d+)', name_splits[1])[1]) + 1
            copied_model.name = f'{name_splits[0]}-copy-{i}'

        return copied_model
