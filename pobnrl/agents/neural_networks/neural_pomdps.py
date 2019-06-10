""" POMDP dynamics as neural networks """

from typing import Tuple
import numpy as np
import tensorflow as tf

from agents.neural_networks import two_layer_q_net
from agents.neural_networks.misc import softmax_sample
from environments import ActionSpace
from misc import tf_run, DiscreteSpace


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

        # feed forward
        self._input_ph = tf.placeholder(
            tf.int32,
            shape=[None, self.state_space.ndim + self.action_space.n],
            name=f"{name}_input"
        )

        self._predict = two_layer_q_net(
            tf.cast(self._input_ph, tf.float32),
            n_out=self._num_state_out + self._num_obs_out,
            n_hidden=conf.network_size,
            scope=f"{name}_net"
        )

        # training data holders
        self._train_new_states_ph = tf.placeholder(
            tf.int32,
            shape=[None, self.state_space.ndim],
            name=f"{name}_new_states"
        )

        self._train_obs_ph = tf.placeholder(
            tf.int32,
            shape=[None, self.obs_space.ndim],
            name=f"{name}_obs"
        )

        # compute losses
        state_losses = [
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._train_new_states_ph[:, i],
                logits=self._predict[
                    :,
                    sum(self.state_space.size[:i]):
                    sum(self.state_space.size[:i + 1])
                ],
                name=f"{name}_state_dim_{i}"
            ) for i in range(self.state_space.ndim)
        ]

        obs_losses = [
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._train_obs_ph[:, i],
                logits=self._predict[
                    :,
                    self._num_state_out + sum(self.obs_space.size[:i]):
                    self._num_state_out + sum(self.obs_space.size[:i + 1])
                ],
                name=f"{name}_obs_dim_{i}"
            ) for i in range(self.obs_space.ndim)
        ]

        self._train_op = tf.train.AdamOptimizer(conf.learning_rate).minimize(
            tf.reduce_mean(tf.stack([*state_losses, *obs_losses], axis=0)),
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"{name}_net")
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

        net_out = tf_run(
            self._predict,
            feed_dict={self._input_ph: np.array([*state, *self.action_space.one_hot(action)])[None]}
        )[0]  # squeeze batch_size dimension

        new_state = [  # softmax sample of each dimension
            softmax_sample(
                net_out[sum(self.state_space.size[:i]):
                        sum(self.state_space.size[:i + 1])]
            )
            for i in range(self.state_space.ndim)
        ]

        obs = [  # softmax sample of each dimension
            softmax_sample(
                net_out[self._num_state_out + sum(self.obs_space.size[:i]):
                        self._num_state_out + sum(self.obs_space.size[:i + 1])]
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

        net_input = np.array([
            [*state, *self.action_space.one_hot(action).astype(int)]
            for state, action in zip(states, actions)
        ])

        tf_run(
            self._train_op,
            feed_dict={
                self._input_ph: net_input,
                self._train_new_states_ph: new_states,
                self._train_obs_ph: obs
            }
        )
