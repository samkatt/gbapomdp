""" POMDP dynamics as neural networks """

from typing import Tuple
import numpy as np
import tensorflow as tf

from agents.neural_networks import simple_fc_nn
from environments import ActionSpace
from misc import DiscreteSpace
from tf_api import tf_run, tf_board_write, tf_writing_to_board


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

        with tf.name_scope(name):

            self._input_states_ph = tf.compat.v1.placeholder(
                tf.int32, shape=[None, self.state_space.ndim], name="input_states"
            )

            self._input_actions_ph = tf.compat.v1.placeholder(
                tf.int32, shape=[None, self.action_space.n], name="input_actions"
            )

            self._input_new_states_ph = tf.compat.v1.placeholder(
                tf.int32, shape=[None, self.state_space.ndim], name="new_state_targets"
            )

            with tf.name_scope("transition_model"):

                input_t = tf.concat(
                    [self._input_states_ph, self._input_actions_ph], axis=1, name='input_T'
                )

                new_state_logits = simple_fc_nn(
                    tf.cast(input_t, tf.float32),
                    n_out=np.sum(self.state_space.size),
                    n_hidden=conf.network_size
                )

                state_losses = [
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self._input_new_states_ph[:, i],
                        logits=new_state_logits[
                            :,
                            sum(self.state_space.size[:i]):
                            sum(self.state_space.size[:i + 1])
                        ],
                        name=f"state_loss_{i}"
                    ) for i in range(self.state_space.ndim)
                ]

                self._sample_states = tf.concat(
                    [
                        tf.random.categorical(
                            new_state_logits[:, sum(state_space.size[:i]):sum(state_space.size[:i + 1])],
                            num_samples=1,
                            name=f'sample_state_feature_{i}')
                        for i in range(state_space.ndim)
                    ],
                    axis=1,
                    name='combine_new_state_features'
                )

            # observation model
            with tf.name_scope("observation_model"):

                input_o = tf.concat(
                    [self._input_states_ph, self._input_actions_ph, self._input_new_states_ph],
                    axis=1,
                    name="input_O"
                )

                observation_logits = simple_fc_nn(
                    tf.cast(input_o, tf.float32),
                    n_out=np.sum(obs_space.size),
                    n_hidden=conf.network_size
                )

                self._train_obs_ph = tf.compat.v1.placeholder(
                    tf.int32,
                    shape=[None, obs_space.ndim],
                    name="observation_targets"
                )

                obs_losses = [
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self._train_obs_ph[:, i],
                        logits=observation_logits[
                            :,
                            sum(obs_space.size[:i]):
                            sum(obs_space.size[:i + 1])
                        ],
                        name=f"observation_loss_{i}"
                    ) for i in range(obs_space.ndim)
                ]

                self._sample_observations = tf.concat(
                    [
                        tf.random.categorical(
                            observation_logits[:, sum(obs_space.size[:i]):sum(obs_space.size[:i + 1])],
                            num_samples=1,
                            name=f'sample_obs_feature_{i}')
                        for i in range(obs_space.ndim)
                    ],
                    axis=1,
                    name='combine_obs_features'
                )

            optimizer = tf.compat.v1.train.AdamOptimizer(conf.learning_rate)

            grads_and_vars = optimizer.compute_gradients(
                tf.reduce_mean(tf.stack([*state_losses, *obs_losses], axis=0)),
                var_list=tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.compat.v1.get_default_graph().get_name_scope()
                ))

            self._train_op = optimizer.apply_gradients(grads_and_vars)

            if tf_writing_to_board(conf):
                self.train_diag = tf.compat.v1.summary.merge(
                    [
                        tf.compat.v1.summary.scalar('obs loss', tf.reduce_mean(obs_losses)),
                        tf.compat.v1.summary.scalar('state loss', tf.reduce_mean(state_losses))
                    ]
                )
            else:
                self.train_diag = tf.no_op('no-diagnostics')

    def simulation_step(self, state: np.array, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """ The simulation step of this dynamics model: S x A -> S, O

        Args:
             state: (`np.array`): input state
             action: (`int`): chosen action

        RETURNS (`Tuple[np.ndarray, np.ndarray]`): [state, observation]

        """

        assert self.state_space.contains(state), f"{state} not in {self.state_space}"
        assert self.action_space.contains(action), f"{action} not in {self.action_space}"

        state = state[None]
        action = self.action_space.one_hot(action)[None]

        new_state = tf_run(
            self._sample_states,
            feed_dict={
                self._input_states_ph: state,
                self._input_actions_ph: action
            }
        )

        observation = tf_run(
            self._sample_observations,
            feed_dict={
                self._input_states_ph: state,
                self._input_actions_ph: action,
                self._input_new_states_ph: new_state
            }
        )

        return new_state[0], observation[0]

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

        actions = [self.action_space.one_hot(a) for a in actions]

        _, diag = tf_run(
            [self._train_op, self.train_diag],
            feed_dict={
                self._input_states_ph: states,
                self._input_actions_ph: actions,
                self._input_new_states_ph: new_states,
                self._train_obs_ph: obs
            }
        )

        tf_board_write(diag)
