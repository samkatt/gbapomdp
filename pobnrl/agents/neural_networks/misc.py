""" Contains miscelaneous functionality for neural networks

Actual networks to compute Q values

ReplayBuffer: stores interactions

Loss calculations

"""

from collections import deque
from typing import Deque, List, Tuple, Any
import random

import tensorflow as tf

import numpy as np


def loss(q_values, targets, loss_type: str):
    """ computes the loss over qval versus targets given configurations

    Returns the loss over Q-values, given their target and type of loss

    Args:
         q_values: q-value estimates
         targets: target q-value estimates
         loss_type: (`str`): is "rmse" or "huber" to what loss to use

    """

    # training operation loss
    if loss_type == "rmse":
        return tf.losses.mean_squared_error(targets, q_values)
    if loss_type == "huber":
        return tf.losses.huber_loss(targets, q_values, delta=10.0)

    raise ValueError('Entered unknown value for loss ' + loss_type)


class ReplayBuffer():
    """ ReplayBuffer stores and samples interactions with the env

    Assumes action is represented by an int
    Uses zeros as padding for observations and -1 for actions

    """

    SIZE = 5000

    def __init__(self, observation_shape: Tuple[int] = None):
        """ constructs the replay buffer for specified observation shape

        Args:
             observation_shape: (`Tuple[int]`): the shape of an observation

        """

        # TODO: remove when completely refactored replay buffer
        if observation_shape is None:
            observation_shape = (1,)

        self._obs_shape = observation_shape
        self._total = 0

        self._obs = np.full((self.SIZE, *self._obs_shape), 'nan', float)
        self._actions = np.full(self.SIZE, -1, int)
        self._rewards = np.full(self.SIZE, 'nan', float)
        self._terminals = np.full(self.SIZE, True, bool)

        self.episodes: Deque[List[Any]] = deque([], self.SIZE)
        self.episodes.append([])

    @property
    def capacity(self) -> int:
        """ returns the total (potential) size of the buffer """
        return self.SIZE

    @property
    def size_2(self) -> int:
        """ returns number of episodes in the buffer

        TODO: rename when done

        """
        return len(self.episodes)

    def store_2(self, step: Any, terminal: bool):
        """ TODO: doc & rename when done """

        self.episodes[-1].append(step)

        if terminal:
            self.episodes.append([])

    @staticmethod
    def sample_episode(episode: List[Any], history_len: int) -> List[Any]:

        end_index = random.randint(1, len(episode))

        start_index = max(0, end_index - history_len)

        return episode[start_index:end_index]

    def sample_2(self, batch_size: int, history_len: int = 1) -> List[List[Any]]:
        """ TODO: rename and doc when done """

        max_sample = self.size_2 - 2

        return [
            self.sample_episode(self.episodes[random.randint(0, max_sample)], history_len)
            for _ in range(batch_size)]

    @property
    def index(self) -> int:
        """ The current index in the replay buffer

        TODO: remove when not used anymore

        """
        return self._total % self.SIZE

    @property
    def max_sample_index(self) -> int:
        """ The maximum index to sample from the buffer

        TODO: remove when not used anymore

        """
        return min(self.SIZE, self._total - 1)

    @property
    def size(self) -> int:
        """ returns the number of samples in the buffer

        RETURNS (`int`):

        """
        return self._total

    def store(  # pylint: disable=too-many-arguments
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            next_obs: np.ndarray,
            terminal: bool) -> None:
        """ stores interaction <obs, action, reward, terminal>

        Args:
        obs: (`np.ndarray`): the 'previous' observation
        action: (`int`): the action taken
        reward: (`float`): the reward after taking action
        next_obs: (`np.ndarray`): the 'next' observation
        terminal: (`bool`): whther the action was terminal

        """

        self.store_2(
            {
                'obs': obs,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'next_obs': next_obs},
            terminal
        )

        self._obs[self.index] = np.copy(obs)
        self._actions[self.index] = action
        self._rewards[self.index] = reward
        self._terminals[self.index] = terminal

        self._total += 1

    # pylint: disable=too-many-locals
    def sample(self, batch_size: int, history_len: int, padding: str) -> dict:
        """ samples batch_size amount

        The argument `padding` determines whether the sequences will be zero'd
        out on the right or left side

        Args:
        batch_size: (`int`): number of batches
        history_len: (`int`): history length to consider
        padding: (`str`): in ['left', 'right']

        RETURNS (`dict`): {'obs': np.ndarray , 'actions': np.ndarray,
        'rewards': np.ndarray, 'terminals: np.ndarray', 'next_obs': np.ndarray,
        'seq_lengths': np.array}

        Where each element in the dictionary is (`batch_size` by `history_len`)
        large amount of elements. The exception is seq_lengths, which describes
        the length of each sequence and is a np.array of size `batch_size`

        """

        assert batch_size > 0, "batch_size must be > 0"
        assert history_len > 0, "history_len must be > 0"
        assert padding in ['left', 'right'], "padding expect 'left' or 'right"

        batch = self.sample_2(batch_size, history_len)

        batch_shape = (batch_size, history_len)

        trace_lengths = np.array([len(trace) for trace in batch])

        reward = np.full(batch_shape, float('nan'))  # nan padding
        terminal = np.full(batch_shape, False)  # False padding
        obs = np.zeros(batch_shape + self._obs_shape)  # 0 padding
        action = np.full(batch_shape, -1)  # -1 padding

        for i, seq in enumerate(batch):
            if padding == 'right':
                for thing in ['obs', 'reward', 'action', 'terminal']:
                    eval(thing)[i][:trace_lengths[i]] = [step[thing] for step in seq]
            else:
                for thing in ['obs', 'reward', 'action', 'terminal']:
                    eval(thing)[i][-trace_lengths[i]:] = [step[thing] for step in seq]

        # construct next observation sequence
        next_ob = [seq[-1]['next_obs'] for seq in batch]

        next_obs = np.concatenate(
            (obs[:, 1:], np.zeros((batch_size, 1) + self._obs_shape)), axis=1
        )

        if padding == 'left':
            # due to padding on the left side, we can simply append the *1*
            # next  observation to the original sequence and remove the first
            next_obs[:, -1] = next_ob

        else:
            # next_obs are being appened where the sequence ended
            next_obs[np.arange(batch_size), trace_lengths - 1] = next_ob

        return {
            'obs': obs,
            'actions': action,
            'rewards': reward,
            'terminals': terminal,
            'next_obs': next_obs,
            'seq_lengths': trace_lengths
        }

    def trace(self, end_index: int, history_len: int) -> np.array:
        """ constructs a trace of interactions ending in end_index

        The trace is at most `history_len` long, but is shorter if one of
        the interactions turned out to be terminating the (previous) episode.

        In that case, it starts at the first interaction of that episode

        Args:
        end_index: (`int`): the end of the trace
        history_len: (`int`): the maximum length of the trace

        RETURNS (`np.array`): indices of the trace

        """
        first_index = end_index - history_len

        # if there is a terminated step in the trace, stop looking back
        first_index = next(
            (i for i in range(end_index - 1, first_index - 1, -1)
             if self._terminals[i]),
            first_index
        ) + 1

        return np.arange(first_index, end_index + 1) % self.SIZE  # incl.

    def clear(self):
        """ clears out all experiences and sets total to 0 """

        self._total = 0

        self._obs[:] = float('nan')
        self._actions[:] = -1
        self._rewards[:] = float('nan')

        # during sample, we do not wish to return sequences that wrap around to
        # the back of the buffer until there is valid data, so we avoid this
        # sampling by setting the last step terminal
        self._terminals[:] = True
