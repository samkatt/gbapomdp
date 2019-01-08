""" DRQN network implementation """

import tensorflow as tf

class DRQNNet:
    """ a network based on DRQN that can return q values and update """

    def __init__(self, env_spaces, rec_arch, optimizer, conf, sess, scope):
        """
        in:

        env_spaces: dict{'O','A'} with observation and action space
        rec_arch: a **Recurrent** QNet
        optimizer: a tf.Optimzer
        conf: configurations (contains observation len,  gamma and loss option)
        sess: the tf session
        scope: name space
        """

        assert rec_arch.is_recursive()

        self.session = sess
        self.rnn_state = None
        self.rec_arch = rec_arch
        self.name = scope

        input_shape = (None, *env_spaces["O"].shape)

        # training operation place holders
        self.obs_t_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        self.obs_tp1_ph = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        # training operation q values and targets
        target_qvalues, _ = self.rec_arch(
            self.obs_tp1_ph,
            env_spaces["A"].n,
            scope=self.name + '_target')

        self.qvalues_op, self.rec_state_op = self.rec_arch(
            self.obs_t_ph,
            env_spaces["A"].n,
            scope=self.name + '_net')

        action_indices = tf.stack([tf.range(tf.size(self.act_t_ph)), self.act_t_ph], axis=-1)
        q_values = tf.gather_nd(self.qvalues_op, action_indices)
        targets = tf.reduce_max(target_qvalues, axis=-1)

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

        net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_net')
        gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=net_vars))

        if conf.clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, 5)

        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # target update operation
        update_target_op = []
        target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name+'_target')

        for var, var_target in zip(sorted(net_vars, key=lambda v: v.name),
                                   sorted(target_vars, key=lambda v: v.name)):
            update_target_op.append(var_target.assign(var))


        self.update_target_op = tf.group(*update_target_op)

        self.session.run(tf.global_variables_initializer())

    def reset(self):
        """ resets the net """
        self.rnn_state = None

    def Qvalues(self, observation):
        """ returns the q values associated with the observations """

        feed_dict = {self.obs_t_ph: observation[None]}

        if self.rnn_state is not None:
            feed_dict[self.rec_arch.rec_state[self.name + "_net"]] = self.rnn_state

        qvals, self.rnn_state = self.session.run(
            [self.qvalues_op, self.rec_state_op],
            feed_dict=feed_dict
        )

        return qvals

    def batch_update(self, obs, actions, rewards, next_obs, done_mask):
        """ performs a batch update """

        self.session.run(self.train_op, feed_dict={
            self.obs_t_ph: obs,
            self.act_t_ph: actions,
            self.rew_t_ph: rewards,
            self.obs_tp1_ph: next_obs,
            self.done_mask_ph: done_mask})

    def update_target(self):
        """ updates the target network """
        self.session.run(self.update_target_op)
