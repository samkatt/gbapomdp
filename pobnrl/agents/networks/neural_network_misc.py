""" Contains miscelaneous functionality for neural networks

Architecture: various network configurations

ReplayBuffer: stores interactions

Misc functions:

    * loss
    * return_estimate

"""

import abc

from tensorflow.python.layers.layers import dense
from tensorflow.python.layers.layers import flatten
import tensorflow as tf

import numpy as np

from misc import sample_n


class Architecture(abc.ABC):
    """ implementation a q-function """

    @abc.abstractmethod
    def __call__(self, net_input, n_actions: int, scope: str):
        """ computes the q values given the net input

        Returns n_actions q values

        Args:
             net_input: the input to the network
             n_actions: (`int`): the number of actions (outputs)
             scope: (`str`): the scope of the network (used by tensorflow)

        """

    @abc.abstractmethod
    def is_recurrent(self) -> bool:
        """ used to check whether the network is recurrent

        Can be useful in determining e.g. whether there is some internal state

        RETURNS (`bool`): whether the network is recurrent

        """


class TwoHiddenLayerQNet(Architecture):
    """ Regular Q network with 2 hidden layers """

    _sizes = {'small': 16, 'med': 64, 'large': 512}

    def __init__(self, network_size: str):
        """ construct the TwoHiddenLayerQNet of the specified size

        Translates 'small' to 16, 'med' to 64 and 'large' to 512 hidden notes

        Args:
             network_size: (`str`): is in {'small', 'med', 'large'}

        """
        assert network_size in ["small", "med", "large"], \
            "network size input invalid"
        self.n_units = self._sizes[network_size]

    def is_recurrent(self) -> bool:
        """ returns false since this network is not recurrent

        RETURNS (`bool`): false (TwoHiddenLayerQNet is not recurrent)

        """
        return False

    def __call__(self, net_input, n_actions: int, scope: str):
        """ returns n_actions Q-values given the network input

        scope must be unique to this network to ensure this works fine
        (tensorflow). This is the main functionality of any network

        Assumes size of input is [batch size, history len, net_input...]

        Args:
             net_input: (tensor) input to the network
             n_actions: (`int`): number of outputs (actions)
             scope: (`str`): the (unique TF) scope of the network

        """
        # concat all inputs but keep batch dimension
        hidden = flatten(net_input)

        # FIXME: programmed without really understanding what is happening
        # it should be possible to call this multiple times
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            print('Using network in scope', scope)
            hidden = dense(
                hidden,
                units=self.n_units,
                activation=tf.nn.tanh,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )
            hidden = dense(
                hidden,
                units=self.n_units,
                activation=tf.nn.tanh,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )

            qvalues = dense(
                hidden,
                units=n_actions,
                activation=None,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )

        return qvalues


class TwoHiddenLayerRecQNet(Architecture):
    """ Recurrent Q network with 2 hidden layers """

    _sizes = {'small': 16, 'med': 64, 'large': 512}
    rec_state = {}  # recurrent state for each scope

    def __init__(self, network_size: int):
        """ construct this of the specified size

        Translates 'small' to 16, 'med' to 64 and 'large' to 512 hidden notes

        Args:
             network_size: (`str`): is in {'small', 'med', 'large'}

        """
        assert network_size in ["small", "med", "large"], \
            "network size input invalid"
        self.n_units = self._sizes[network_size]

    def is_recurrent(self) -> bool:
        """ returns true

        interface implementation of Architecture

        RETURNS (`bool`): true, as this is recurrent

        """
        return True

    def __call__(self, net_input, n_actions: int, scope: str):
        """ returns n_actions Q-values given the network input

        This is the main functionality of any network scope must be unique to
        this network to ensure this works fine (tensorflow).

        Assumes size of input is [batch size, history len, net_input...]

        Args:
             net_input: (tensor) input to the network
             n_actions: (`int`): number of outputs (actions)
             scope: (`str`): the (unique TF) scope of the network

        """
        assert len(net_input.shape) > 2

        batch_size = tf.shape(net_input)[0]
        history_len = tf.shape(net_input)[1]
        observation_num = net_input.shape[2:].num_elements()

        hidden = tf.reshape(  # flatten but keep batch size and history len
            net_input,
            [batch_size, history_len, observation_num]
        )

        # FIXME: programmed without really understanding what is happening
        # it should be possible to call this multiple times
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            print('Using network in scope', scope)
            hidden = dense(
                hidden,
                units=self.n_units,
                activation=tf.nn.tanh,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )
            hidden = dense(
                hidden,
                units=self.n_units,
                activation=tf.nn.tanh,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )

            rnn_cell = tf.nn.rnn_cell.LSTMCell(
                self.n_units,
                initializer=tf.glorot_normal_initializer()
            )

            # can be initialized with a feed dict if you want to set this to a
            # previous state
            self.rec_state[scope] = rnn_cell.zero_state(batch_size, tf.float32)

            # will automatically handle the history len of each batch as a
            # single sequence
            hidden, new_rec_state = tf.nn.dynamic_rnn(
                rnn_cell,
                inputs=hidden,
                initial_state=self.rec_state[scope]
            )

            qvalues = dense(
                hidden[:, -1],
                units=n_actions,
                activation=None,
                bias_initializer=tf.glorot_normal_initializer(),
                kernel_initializer=tf.glorot_normal_initializer()
            )

        return qvalues, new_rec_state


class ReplayBuffer():
    """This is a memory efficient implementation of the replay buffer.

    The sepecific memory optimizations use here are:
        - only store each frame once rather than k times
          even if every observation normally consists of k last frames
        - store frames as np.uint8 (actually it is most time-performance
                to cast them back to float32 on GPU to minimize memory transfer
                time)
        - store frame_t and frame_(t+1) in the same buffer.

    For the tipical use case in Atari Deep RL buffer with 1M frames the total
    memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

    Warning! Assumes that returning frame of zeros at the beginning
    of the episode, when there is less frames than `frame_history_len`,
    is acceptable.
    """

    def __init__(self, size, history_len, use_float):
        """
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.history_len = history_len
        self.obs_dtype = (np.float32 if use_float else np.uint8)

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        """ Returns true if batch_size can be sampled

        `batch_size` different transitions can be sampled from the buffer

        """
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate(
            [self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate(
            [self._encode_observation(idx + 1)[None] for idx in idxes],
            0)

        done_mask = np.array(
            [1.0 if self.done[idx] else 0.0 for idx in idxes],
            dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n(
            lambda: np.random.randint(
                0,
                self.num_in_buffer - 2),
            batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):

        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if self.history_len == 1:
            return np.array([self.obs[idx]])

        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - self.history_len
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        # do not look further in the past than the current episode
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0])
                      for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.array(frames)

        # else:
        return self.obs[start_idx:end_idx]

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs = np.empty([self.size] +
                                list(frame.shape), dtype=self.obs_dtype)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects

        Effects are those of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call
        `encode_recent_observation` in between.

        Paramters
        ---------
        idx: int
            Index in buffer of prev observed frame (returned by `store_frame`)
        action: int
            Action that was performed upon observing this frame
        reward: float
            Reward that was received when the actions was performed
        done: bool
            True if episode was finished after performing that action
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done


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
