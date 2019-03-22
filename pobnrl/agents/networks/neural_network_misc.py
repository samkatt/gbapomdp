""" Contains miscelaneous functionality for neural networks

Actual networks to compute Q values

ReplayBuffer: stores interactions

Misc functions:

    * loss
    * return_estimate

"""

from tensorflow.python.layers.layers import dense
from tensorflow.python.layers.layers import flatten
import tensorflow as tf

import numpy as np


def return_estimate(next_q, next_target, use_double_q: bool):
    """ compute estimated return

    Either returns the max of the target network
    or the double-q estimate (depending on use_double_q)

    Args:
         next_q: the next q-values
         next_target: the target q-values
         use_double_q: (`bool`): whether to use 'double-q' technique

    """

    if not use_double_q:
        return tf.reduce_max(next_target, axis=-1)

    # double_q
    best_action = tf.argmax(next_q, axis=1, output_type=tf.int32)
    best_action_indices = tf.stack(
        [tf.range(tf.size(best_action)), best_action], axis=-1)

    return tf.gather_nd(next_target, best_action_indices)


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


def two_layer_q_net(net_input, n_actions: int, n_units: int, scope: str):
    """ Returns Q-values of input using a two-hidden layer architecture

    TODO: add actions here

    scope must be unique to this network to ensure this works fine
    (tensorflow).

    Assumes size of input is [batch size, history len, net_input...]

    Args:
         net_input: the input of the network (observation)
         n_actions: (`int`): # of actions
         n_units: (`int`): # of units per layer
         scope: (`str`): scope (unique, for tensorflow)

    """
    # concat all inputs but keep batch dimension
    hidden = flatten(net_input)

    # TODO: programmed without really understanding what is happening
    # it should be possible to call this multiple times
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        for layer in range(2):  # 2 hidden layers
            hidden = dense(
                hidden,
                units=n_units,
                activation=tf.nn.tanh,
                name=scope + '_hidden_' + str(layer)
            )

        qvalues = dense(
            hidden,
            units=n_actions,
            activation=None,
            name=scope + '_out'
        )

    return qvalues


def two_layer_rec_q_net(  # pylint: disable=too-many-arguments
        net_input,
        seq_lengths,
        state,
        n_actions: int,
        n_units: int,
        scope: str):
    """ Returns Q-values of input using a two-hidden (rec) layer architecture

    TODO: add actions here

    scope must be unique to this network to ensure this works fine
    (tensorflow).

    Assumes size of input is [batch size, history len, net_input...]

    Args:
         net_input: the input of the network (observation)
         seq_lengths: the length of each batch
         state: state of the recurrent layer
         n_actions: (`int`): # of actions
         n_units: (`int`): # of units per layer
         scope: (`str`): scope (unique, for tensorflow)

    """

    assert len(net_input.shape) > 2

    batch_size = tf.shape(net_input)[0]
    history_len = tf.shape(net_input)[1]
    observation_num = net_input.shape[2:].num_elements()

    # flatten but keep batch size and history len
    hidden = tf.reshape(
        net_input,
        [batch_size, history_len, observation_num]
    )

    # TODO: programmed without really understanding what is happening
    # it should be possible to call this multiple times
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        for layer in range(2):  # 2 hidden layers
            hidden = dense(
                hidden,
                units=n_units,
                activation=tf.nn.tanh,
                name=scope + '_hidden_' + str(layer)
            )

        rnn_cell = tf.nn.rnn_cell.LSTMCell(n_units)
        state = rnn_cell.zero_state(batch_size, tf.float32)

        # handlse the history len of each batch as a single sequence
        hidden, new_rec_state = tf.nn.dynamic_rnn(
            rnn_cell,
            sequence_length=seq_lengths,
            inputs=hidden,
            initial_state=state,
            dtype=tf.float32,
            scope=scope + '_rnn'
        )

        seq_q_mask = tf.stack(
            [tf.range(tf.size(seq_lengths)), seq_lengths - 1],
            axis=-1
        )

        qvalues = dense(
            tf.gather_nd(hidden, seq_q_mask),
            units=n_actions,
            activation=None,
            name=scope + '_out'
        )

    return qvalues, new_rec_state


class ReplayBuffer():
    """ ReplayBuffer stores and samples interactions with the env

    Assumes action is represented by an int
    Uses zeros as padding for observations and -1 for actions

    """

    SIZE = 100000

    def __init__(self, observation_shape: tuple):
        """ constructs the replay buffer for specified observation shape

        Args:
             observation_shape: (`tuple`): the shape of an observation

        """

        self._obs_shape = observation_shape
        self._total = 0

        self._obs = np.full((self.SIZE, *self._obs_shape), 'nan', float)
        self._actions = np.full(self.SIZE, -1, int)
        self._rewards = np.full(self.SIZE, 'nan', float)
        self._terminals = np.full(self.SIZE, None, bool)

        # during sample, we do not wish to return sequences that wrap around to
        # the back of the buffer until there is valid data, so we avoid this
        # sampling by setting the last step terminal
        self._terminals[-1] = True

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
                # [0,0,0,1,1,1]
                return np.concatenate(
                    [np.zeros(history_len - seq_len), np.ones(seq_len)]
                )
        else:
            def trace_mask(seq_len):
                # [1,1,1,0,0,0]
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

        obs = np.full(batch_shape + self._obs_shape, .0)
        obs[sample_mask] = self._obs[trace_indices]

        actions = np.full(batch_shape, -1)  # -1 padding
        actions[sample_mask] = self._actions[trace_indices]

        # construct next observation sequence
        next_ob = self._obs[(sample_indices + 1) % self.SIZE]

        if padding == 'left':
            # due to padding on the left side, we can simply append the *1*
            # next  observation to the original sequence and remove the first
            next_obs = np.concatenate((obs[:, 1:], next_ob[:, None]), axis=1)

        else:
            next_obs = np.zeros(obs.shape)
            # when padding is left, we do not want to simply append the
            # next observation at the end: if the sequence ended before the
            # max len, then we must place the next observation where the
            # sequence ended
            complete_sequences = (trace_lengths == history_len)

            next_obs[~complete_sequences] = obs[~complete_sequences]
            next_obs[~complete_sequences, trace_lengths[~complete_sequences]] \
                = next_ob[~complete_sequences]

            next_obs[complete_sequences, :-1] = obs[complete_sequences, 1:]
            next_obs[complete_sequences, -1] = next_ob[complete_sequences]

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
