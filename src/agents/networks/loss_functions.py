""" defines loss functions """

import tensorflow as tf

def return_estimate(next_q, next_target, conf):
    """ expected next return

    Either returns the max of the target network
    or the double-q estimate

    Assumes: conf.double_q is a bool
    """

    if not conf.double_q:
        return tf.reduce_max(next_target, axis=-1)
    else: # double_q
        best_action = tf.argmax(next_q, axis=1, output_type=tf.int32)
        best_action_indices = tf.stack([tf.range(tf.size(best_action)), best_action], axis=-1)
        return tf.gather_nd(next_target, best_action_indices)

def loss(q_values, targets, conf):
    """ returns the loss over qval versus targets given configurations

    Assumes: conf.loss being rmse or huber
    """

    # training operation loss
    if conf.loss == "rmse":
        return tf.losses.mean_squared_error(targets, q_values)
    elif conf.loss == "huber":
        return tf.losses.huber_loss(targets, q_values, delta=10.0)
    else:
        raise ValueError('Entered unknown value for loss ' + conf.loss)
