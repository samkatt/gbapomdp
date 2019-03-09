""" DQN network implementation """

import tensorflow as tf

from agents.networks.qnets.QNetInterface import QNetInterface
import agents.networks.architectures
import agents.networks.architectures as archs
import agents.networks.loss_functions as loss_functions
import utils.tf_wrapper as tf_wrapper


class DQNNet(QNetInterface):
    """ a network based on DQN that can return q values and update """

    # FIXME: take specific arguments instead of conf
    def __init__(
            self,
            env_spaces: dict,
            arch: agents.networks.architectures.Architecture,
            optimizer,
            conf,
            scope: str):
        """ construct the DRQNNet

        Assumes the input architecture arch is **not** recurrent

        Args:
             env_spaces: (`dict`): {'O','A'} with observation and action space
             rec_arch: (`pobnrl.agents.networks.architectures.Architecture`): the architecture
             optimizer: the type of optimizer to use for learning (tf.optimizer)
             conf: configuration file
             scope: (`str`): name space of the network

        """

        assert not arch.is_recurrent()

        input_shape = (conf.observation_len, *env_spaces["O"].shape)

        # training operation place holders
        self.obs_t_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        self.obs_tp1_ph = tf.placeholder(
            tf.float32, [None] + list(input_shape))
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        # define operations to retrieve q and target values
        self.qvalues_fn = arch(
            self.obs_t_ph,
            env_spaces["A"].n,
            scope=scope + '_net'
        )

        next_qvalues_fn = arch(
            self.obs_tp1_ph,
            env_spaces["A"].n,
            scope=scope + '_net'
        )

        next_targets_fn = arch(
            self.obs_tp1_ph,
            env_spaces["A"].n,
            scope=scope + '_target'
        )

        if conf.random_priors:  # add random function to our estimates
            prior = archs.TwoHiddenLayerQNet(conf.network_size)
            prior_vals = prior(
                self.obs_t_ph,
                env_spaces["A"].n,
                scope=scope + '_prior'
            )

            next_prior_vals = prior(
                self.obs_tp1_ph,
                env_spaces["A"].n,
                scope=scope + '_prior'
            )

            self.qvalues_fn = tf.add(self.qvalues_fn, prior_vals)

            next_targets_fn = tf.add(next_targets_fn, next_prior_vals)

        # define loss
        action_onehot = tf.stack(
            [tf.range(tf.size(self.act_t_ph)), self.act_t_ph], axis=-1)
        q_values = tf.gather_nd(self.qvalues_fn, action_onehot)

        return_estimate = loss_functions.return_estimate(
            next_qvalues_fn,
            next_targets_fn,
            conf.double_q
        )

        targets = tf.where(
            tf.cast(self.done_mask_ph, tf.bool),
            x=self.rew_t_ph, y=self.rew_t_ph + (conf.gamma * return_estimate))

        loss = loss_functions.loss(q_values, targets, conf.loss)

        net_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope + '_net')
        gradients, variables = zip(
            *optimizer.compute_gradients(loss, var_list=net_vars))

        if conf.clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, 5)

        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # target update operation
        update_target_op = []
        target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope + '_target')

        for var, var_target in zip(sorted(net_vars, key=lambda v: v.name),
                                   sorted(target_vars, key=lambda v: v.name)):
            update_target_op.append(var_target.assign(var))

        self.update_target_op = tf.group(*update_target_op)

        tf_wrapper.get_session().run(tf.global_variables_initializer())

    def reset(self):
        """ no internal state so does nothing, interface requirement """

    def is_recurrent(self) -> bool:  # pylint: disable=no-self-use
        """ False: the DQNNet is not recurrent

        interface requirement

        RETURNS (`bool`): False

        """
        return False

    def qvalues(self, observation):
        """ returns the Q-values associated with the observation (net input)

        Args:
             observation: the input to the network

        """

        return tf_wrapper.get_session().run(
            self.qvalues_fn,
            feed_dict={self.obs_t_ph: observation[None]}
        )

    def batch_update(self, obs, actions, rewards, next_obs, done_mask):
        """ performs a batch update on the network on the provided input

        Basically the stochastic gradient descent step where, for any i,
        obs[i], actions[i], rewards[i], next_obs[i] and done_mask[i] are respectively the
        observation, actions, reward, next observation and terminality of some step i

        Args:
             obs: the observations or input to the network
             actions: the chosen action
             rewards: the rewards associated with each obs-action pair
             next_obs: the next observation after taking action and seeing obs
             done_mask: a boolean for each transition showing whether the transition was terminal

        """

        tf_wrapper.get_session().run(self.train_op, feed_dict={
            self.obs_t_ph: obs,
            self.act_t_ph: actions,
            self.rew_t_ph: rewards,
            self.obs_tp1_ph: next_obs,
            self.done_mask_ph: done_mask})

    def update_target(self):
        """ updates the target network """
        tf_wrapper.get_session().run(self.update_target_op)
