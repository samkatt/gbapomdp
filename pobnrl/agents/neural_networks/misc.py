""" Contains miscelaneous functionality for neural networks

Actual networks to compute Q values

ReplayBuffer: stores interactions

Loss calculations

"""

import random

from collections import deque
from typing import Deque, List, Any

import torch
from torch.nn.modules.loss import _Loss as TorchLoss


class RMSELoss(TorchLoss):  # type: ignore
    """ custom RMSELoss criterion in pytorch """

    def forward(
            self,
            prediction: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        """ forward pass, returns loss of rmse(prediction, target)

        Args:
             prediction: (`torch.Tensor`):
             target: (`torch.Tensor`):

        """
        return torch.sqrt(
            torch.nn.functional.mse_loss(prediction, target, reduction=self.reduction)
        )


def loss_criterion(loss_type: str) -> TorchLoss:
    """ factory for pytorch loss criterions

    Args:
         loss_type: (`str`): in ['rmse', 'huber']

    RETURNS (`torch.nn.Criterion`):

    """

    if loss_type == "rmse":
        return RMSELoss()
    if loss_type == "huber":
        return torch.nn.SmoothL1Loss()

    raise ValueError('Entered unknown value for loss ' + loss_type)


class ReplayBuffer():
    """ ReplayBuffer stores and samples interactions with the env

    Assumes action is represented by an int
    Uses zeros as padding for observations and -1 for actions

    """

    SIZE = 5000

    def __init__(self) -> None:
        """ constructs the replay buffer """

        self.episodes: Deque[List[Any]] = deque([], self.SIZE)
        self.episodes.append([])

    @property
    def capacity(self) -> int:
        """ returns the total (potential) size of the buffer

        RETURNS (`int`):

        """
        return self.SIZE

    @property
    def size(self) -> int:
        """ returns number of episodes in the buffer

        RETURNS (`int`):

        """
        return len(self.episodes)

    def store(self, step: Any, terminal: bool):
        """ stores step in the buffer

        Step can be anything, the buffer does not care, but it will be appended
        to the current episode. When the replay buffer is sampled, `step` may
        be returned stochastically. If `terminal` is True, then a new episode
        will start after this sample

        Args:
             step: (`Any`): whatever must be stored in the current time step
             terminal: (`bool`): whether the step is terminal

        """

        self.episodes[-1].append(step)

        if terminal:
            self.episodes.append([])

    @staticmethod
    def sub_sample_episode(episode: List[Any], history_len: int) -> List[Any]:
        """ samples a trace of max `history_len` from the episode

        Args:
             episode: (`List[Any]`): a list of time steps
             history_len: (`int`): the max length of the returned sample

        RETURNS (`List[Any]`):

        """

        end_index = random.randint(1, len(episode))

        start_index = max(0, end_index - history_len)

        return episode[start_index:end_index]

    def sample(self, batch_size: int, history_len: int = 1) -> List[List[Any]]:
        """ samples from the replay buffer

        Will sample a batch_size by history_len list of time steps

        Args:
             batch_size: (`int`): the number of sub episodes to sample
             history_len: (`int`): the (max) length of a sub episode

        RETURNS (`List[List[Any]]`): batch_size x history_len

        """

        assert batch_size > 0, "batch_size must be > 0"
        assert history_len > 0, "history_len must be > 0"

        max_sample = self.size - 2

        return [
            self.sub_sample_episode(self.episodes[random.randint(0, max_sample)], history_len)
            for _ in range(batch_size)
        ]

    def clear(self) -> None:
        """ clears out all experiences and sets total to 0 """
        self.episodes.clear()
        self.episodes.append([])
