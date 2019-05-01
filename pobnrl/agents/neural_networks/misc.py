""" Contains miscelaneous functionality for neural networks

Actual networks to compute Q values

ReplayBuffer: stores interactions

Loss calculations

"""

from typing import Tuple

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

    SIZE = 100000

    def __init__(self, observation_shape: Tuple[int]):
        """ constructs the replay buffer for specified observation shape

        Args:
             observation_shape: (`Tuple[int]`): the shape of an observation

        """

        self._obs_shape = observation_shape
        self._total = 0

        self._obs = np.full((self.SIZE, *self._obs_shape), 'nan', float)
        self._actions = np.full(self.SIZE, -1, int)
        self._rewards = np.full(self.SIZE, 'nan', float)
        self._terminals = np.full(self.SIZE, True, bool)

    @property
    def index(self) -> int:
        """ The current index in the replay buffer """
        return self._total % self.SIZE

    @property
    def max_sample_index(self) -> int:
        """ The maximum index to sample from the buffer """
        return min(self.SIZE, self._total - 1)

    @property
    def size(self) -> int:
        """ returns the number of samples in the buffer

        RETURNS (`int`):

        """
        return self._total

    def store(
            self,
            obs: np.array,
            action: int,
            reward: float,
            terminal: bool) -> None:
        """ stores interaction <obs, action, reward, terminal>

        Args:
        obs: (`np.array`): the 'previous' observation
        action: (`int`): the action taken
        reward: (`float`): the reward after taking action
        terminal: (`bool`): whther the action was terminal

        """

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

        batch_shape = (batch_size, history_len)

        sample_indices = np.random.randint(
            0, self.max_sample_index, batch_size
        )

        sample_traces = [self.trace(i, history_len) for i in sample_indices]
        trace_indices = np.concatenate(sample_traces)
        trace_lengths = np.array([len(t) for t in sample_traces])

        if padding == 'left':
            def trace_mask(seq_len):
                # [0,0,0,t1,t2,t3]
                return np.concatenate(
                    [np.zeros(history_len - seq_len), np.ones(seq_len)]
                )
        else:
            def trace_mask(seq_len):
                # [t1,t2,t3,0,0,0]
                return np.concatenate(
                    [np.ones(seq_len), np.zeros(history_len - seq_len)]
                )

        # True/False mask of batch_shape to pick the values to be changed
        sample_mask = np.array(
            [trace_mask(l) for l in trace_lengths],
            dtype=bool
        )

        # construct results by first initializing them with default
        # values (acting as padding where necessary) and then fill in
        # the values with the traces

        rewards = np.full(batch_shape, float('nan'))  # nan padding
        rewards[sample_mask] = self._rewards[trace_indices]

        terminals = np.full(batch_shape, False)  # False padding
        terminals[sample_mask] = self._terminals[trace_indices]

        obs = np.zeros(batch_shape + self._obs_shape)  # 0 padding
        obs[sample_mask] = self._obs[trace_indices]

        actions = np.full(batch_shape, -1)  # -1 padding
        actions[sample_mask] = self._actions[trace_indices]

        # construct next observation sequence
        next_ob = self._obs[(sample_indices + 1) % self.SIZE]

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
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
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
