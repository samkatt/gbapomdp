""" POMDP dynamics as neural networks """

from typing import Tuple
import numpy as np
import tensorflow as tf

from agents.neural_networks import simple_fc_nn
from agents.neural_networks.misc import softmax_sample
from environments import ActionSpace
from misc import DiscreteSpace
from tf_api import tf_run, tf_board_write


class DynamicsModel():
    """ A neural network representing POMDP dynamics (s,a) -> p(s',o) """

    def __init__(
            self,
            state_space: DiscreteSpace,
            action_space: ActionSpace,
            obs_space: DiscreteSpace,
            conf,
            name: str):
        """ Creates a dynamic model

        Args:
             state_space: (`pobnrl.misc.DiscreteSpace`):
             action_space: (`pobnrl.environments.ActionSpace`):
             obs_space: (`pobnrl.misc.DiscreteSpace`):
             conf: configurations from program input
             name: (`str`): name of this model (unique)

        """

        self.state_space = state_space
        self.action_space = action_space
        self.obs_space = obs_space

        self._num_state_out = np.sum(self.state_space.size)
        self._num_obs_out = np.sum(self.obs_space.size)

        with tf.name_scope(name):
            # feed forward
            self._input_t = tf.placeholder(
                tf.int32,
                shape=[None, self.state_space.ndim + self.action_space.n],
                name="input_T"
            )

            self._input_o = tf.placeholder(
                tf.int32,
                shape=[None, 2 * self.state_space.ndim + self.action_space.n],
                name="input_O"
            )

            with tf.name_scope("T"):
                self._new_state_logits = simple_fc_nn(
                    tf.cast(self._input_t, tf.float32),
                    n_out=self._num_state_out,
                    n_hidden=conf.network_size
                )

            with tf.name_scope("O"):
                self._observation_logits = simple_fc_nn(
                    tf.cast(self._input_o, tf.float32),
                    n_out=self._num_obs_out,
                    n_hidden=conf.network_size
                )

            # training data holders
            self._train_new_states_ph = tf.placeholder(
                tf.int32,
                shape=[None, self.state_space.ndim],
                name="new_state_logits"
            )

            self._train_obs_ph = tf.placeholder(
                tf.int32,
                shape=[None, self.obs_space.ndim],
                name="obs_logits"
            )

            # compute losses
            state_losses = [
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self._train_new_states_ph[:, i],
                    logits=self._new_state_logits[
                        :,
                        sum(self.state_space.size[:i]):
                        sum(self.state_space.size[:i + 1])
                    ],
                    name=f"state_loss_{i}"
                ) for i in range(self.state_space.ndim)
            ]

            obs_losses = [
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self._train_obs_ph[:, i],
                    logits=self._observation_logits[
                        :,
                        sum(self.obs_space.size[:i]):
                        sum(self.obs_space.size[:i + 1])
                    ],
                    name=f"obs_loss_{i}"
                ) for i in range(self.obs_space.ndim)
            ]

            self.train_diag = tf.summary.merge([
                tf.summary.scalar('obs loss', tf.reduce_mean(obs_losses)),
                tf.summary.scalar('state loss', tf.reduce_mean(state_losses))
            ])

            self._train_op = tf.train.AdamOptimizer(conf.learning_rate).minimize(
                tf.reduce_mean(tf.stack([*state_losses, *obs_losses], axis=0)),
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=f"{tf.get_default_graph().get_name_scope()}/T"
                ) + tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=f"{tf.get_default_graph().get_name_scope()}/O"
                )
            )

    def simulation_step(self, state: np.array, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """ The simulation step of this dynamics model: S x A -> S, O

        Args:
             state: (`np.array`): input state
             action: (`int`): chosen action

        RETURNS (`Tuple[np.ndarray, np.ndarray]`): [state, observation]

        """

        assert self.state_space.contains(state), f"{state} not in {self.state_space}"
        assert self.action_space.contains(action), f"{action} not in {self.action_space}"

        new_state_logits = tf_run(
            self._new_state_logits,
            feed_dict={self._input_t: np.array([*state, *self.action_space.one_hot(action)])[None]}
        )[0]  # squeeze batch_size dimension

        new_state = [  # softmax sample of each dimension
            softmax_sample(
                new_state_logits[sum(self.state_space.size[:i]):
                                 sum(self.state_space.size[:i + 1])]
            )
            for i in range(self.state_space.ndim)
        ]

        observation_logits = tf_run(
            self._observation_logits,
            feed_dict={self._input_o: np.array([
                *state,
                *self.action_space.one_hot(action),
                *new_state
            ])[None]}
        )[0]  # squeeze batch_size dim

        obs = [  # softmax sample of each dimension
            softmax_sample(
                observation_logits[sum(self.obs_space.size[:i]):
                                   sum(self.obs_space.size[:i + 1])]
            )
            for i in range(self.obs_space.ndim)
        ]

        return np.array(new_state), np.array(obs)

    def batch_update(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            new_states: np.ndarray,
            obs: np.ndarray) -> None:
        """ performs a batch update (single gradient descent step)

        Args:
             states: (`np.ndarray`): (batch_size, state_shape) array of states
             actions: (`np.ndarray`): (batch_size,) array of actions
             new_states: (`np.ndarray`): (batch_size, state_shape) array of (next) states
             obs: (`np.ndarray`): (batch_size, obs_shape) array of observations

        """

        input_t = np.array([
            [*state, *self.action_space.one_hot(action).astype(int)]
            for state, action in zip(states, actions)
        ])

        input_o = np.array([
            [*state, *self.action_space.one_hot(action).astype(int), *new_state]
            for state, action, new_state in zip(states, actions, new_states)
        ])

        _, diag = tf_run(
            [self._train_op, self.train_diag],
            feed_dict={
                self._input_t: input_t,
                self._input_o: input_o,
                self._train_new_states_ph: new_states,
                self._train_obs_ph: obs
            }
        )

        tf_board_write(diag)
