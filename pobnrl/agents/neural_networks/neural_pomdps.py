""" POMDP dynamics as neural networks """

from typing import Tuple
import numpy as np
import tensorflow as tf

from agents.neural_networks import two_layer_q_net
from environments import ActionSpace
from misc import tf_run, DiscreteSpace


class DynamicsModel():  # pylint: disable=too-many-instance-attributes
    """ A neural network representing POMDP dynamics (s,a) -> p(s',o)

    TODO: optionally use simple indices ( check with Andrea about encodings )

    """

    @staticmethod
    def softmax_sample(arr: np.array) -> int:
        """ returns a soft(arg)max sample from `arr`

        TODO: move somewhere else

        Args:
             arr: (`np.array`): a 1-dimensional numpy array to sample from

        RETURNS (`int`): between 0 ... len(arr)

        """
        ar_softmax = np.exp(arr - arr.max())
        return np.random.choice(len(arr), p=ar_softmax / ar_softmax.sum())

    def __init__(  # pylint: disable=too-many-arguments
            self,
            state_space: DiscreteSpace,
            action_space: ActionSpace,
            obs_space: DiscreteSpace,
            network_size: int,
            optimizer: tf.train.Optimizer,
            name: str):

        self.state_space = state_space
        self.action_space = action_space
        self.obs_space = obs_space
        network_size = network_size

        self._num_state_out = np.sum(self.state_space.size)
        self._num_obs_out = np.sum(self.obs_space.size)

        # feed forward
        self._input_ph = tf.placeholder(
            tf.int32,
            shape=[None, self.state_space.ndim + self.action_space.n],
            name=name + "_input"
        )

        self._predict = two_layer_q_net(
            tf.cast(self._input_ph, tf.float32),
            n_out=self._num_state_out + self._num_obs_out,
            n_hidden=network_size,
            scope=name + "_net"
        )

        # training data holders
        self._train_new_states_ph = tf.placeholder(
            tf.int32,
            shape=[None, self.state_space.ndim],
            name=name + "_new_states"
        )

        self._train_obs_ph = tf.placeholder(
            tf.int32,
            shape=[None, self.obs_space.ndim],
            name=name + "_obs"
        )

        # training losses
        state_losses = [
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._train_new_states_ph[:, i],
                logits=self._predict[
                    :,
                    sum(self.state_space.size[:i]):
                    sum(self.state_space.size[:i + 1])
                ],
                name=name + "_state_dim" + str(i)
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
                name=name + "_obs_dim" + str(i)
            ) for i in range(self.obs_space.ndim)
        ]

        # combine loss
        self._train_op = optimizer.minimize(
            tf.reduce_mean(tf.stack([*state_losses, *obs_losses], axis=0)),
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + "_net")
        )

    def simulation_step(self, state: np.array, action: int) -> Tuple[str, np.ndarray]:

        assert self.state_space.contains(state), f"{state} not in {self.state_space}"
        assert self.action_space.contains(action), f"{action} not in {self.action_space}"

        net_out = tf_run(
            self._predict,
            feed_dict={self._input_ph: np.array([*state, *self.action_space.one_hot(action)])[None]}
        )[0]  # squeeze batch_size dimension

        new_state = [  # softmax sample of each dimension
            self.softmax_sample(
                net_out[sum(self.state_space.size[:i]):
                        sum(self.state_space.size[:i + 1])]
            )
            for i in range(self.state_space.ndim)
        ]

        obs = [  # softmax sample of each dimension
            self.softmax_sample(
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
            obs: np.ndarray):

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
