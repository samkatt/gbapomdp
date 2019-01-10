""" DQN network implementation """

import tensorflow as tf
import utils.tf_wrapper as tf_wrapper

from networks.QNetInterface import QNetInterface

class DQNNet(QNetInterface):
    """ a network based on DQN that can return q values and update """

    def reset(self):
        """ no internal state """
        pass

    def is_recurrent(self):
        return False

    def __init__(self, env_spaces, arch, optimizer, conf, scope):
        """
        in:

        env_spaces: dict{'O','A'} with observation and action space
        arch: a QNet
        optimizer: a tf.Optimzer
        conf: configurations (contains observation len,  gamma and loss option)
        scope: name space
        """

        assert not arch.is_recurrent()

        input_shape = (conf.observation_len, *env_spaces["O"].shape)

        # training operation place holders
        self.obs_t_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        self.obs_tp1_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        # training operation q values and targets
        self.qvalues_op = arch(
            self.obs_t_ph,
            env_spaces["A"].n,
            scope=scope + '_net')

        target_qvalues_op = arch(
            self.obs_tp1_ph,
            env_spaces["A"].n,
            scope=scope + '_target')

        action_indices = tf.stack([tf.range(tf.size(self.act_t_ph)), self.act_t_ph], axis=-1)
        q_values = tf.gather_nd(self.qvalues_op, action_indices)

        if not conf.double_q:
            targets = tf.reduce_max(target_qvalues_op, axis=-1)
        else: # double_q
            double_q_target_qvalues = arch(
                self.obs_tp1_ph,
                env_spaces["A"].n,
                scope=scope + '_net'
                )

            best_action = tf.argmax(double_q_target_qvalues, axis=1, output_type=tf.int32)
            best_action_indices = tf.stack([tf.range(tf.size(best_action)), best_action], axis=-1)
            targets = tf.gather_nd(target_qvalues_op, best_action_indices)


        targets = tf.where(
            tf.cast(self.done_mask_ph, tf.bool),
            x=self.rew_t_ph, y=self.rew_t_ph + (conf.gamma * targets))

        # training operation loss
        if conf.loss == "rmse":
            loss = tf.losses.mean_squared_error(targets, q_values)
        elif conf.loss == "huber":
            loss = tf.losses.huber_loss(targets, q_values, delta=10.0)
        else:
            raise ValueError('Entered unknown value for loss ' + conf.loss)

        net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'_net')
        gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=net_vars))

        if conf.clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, 5)

        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # target update operation
        update_target_op = []
        target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope+'_target')

        for var, var_target in zip(sorted(net_vars, key=lambda v: v.name),
                                   sorted(target_vars, key=lambda v: v.name)):
            update_target_op.append(var_target.assign(var))


        self.update_target_op = tf.group(*update_target_op)

        tf_wrapper.get_session().run(tf.global_variables_initializer())


    def qvalues(self, observation):
        """ returns the q values associated with the observations """

        return tf_wrapper.get_session().run(
            self.qvalues_op,
            feed_dict={self.obs_t_ph: observation[None]}
        )

    def batch_update(self, obs, actions, rewards, next_obs, done_mask):
        """ performs a batch update """

        tf_wrapper.get_session().run(self.train_op, feed_dict={
            self.obs_t_ph: obs,
            self.act_t_ph: actions,
            self.rew_t_ph: rewards,
            self.obs_tp1_ph: next_obs,
            self.done_mask_ph: done_mask})

    def update_target(self):
        """ updates the target network """
        tf_wrapper.get_session().run(self.update_target_op)
