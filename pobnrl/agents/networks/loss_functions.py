""" defines loss functions """

import tensorflow as tf


def return_estimate(next_q, next_target, use_double_q: bool):
    """ expected next return

    Either returns the max of the target network
    or the double-q estimate (depending on use_double_q)

    Assumes: use_double_q is a bool

    :param next_q:
    :param next_target:
    :param use_double_q:
    :type use_double_q: bool whether to use double q or not
    """

    if not use_double_q:
        return tf.reduce_max(next_target, axis=-1)

    # double_q
    best_action = tf.argmax(next_q, axis=1, output_type=tf.int32)
    best_action_indices = tf.stack(
        [tf.range(tf.size(best_action)), best_action], axis=-1)

    return tf.gather_nd(next_target, best_action_indices)


def loss(q_values, targets, loss_type: str):
    """ returns the loss over qval versus targets given configurations

    Assumes: conf.loss being "rmse" or "huber"

    :param q_values:
    :param targets:
    :param loss_type: str which type of loss to use ()
    """

    # training operation loss
    if loss_type == "rmse":
        return tf.losses.mean_squared_error(targets, q_values)
    if loss_type == "huber":
        return tf.losses.huber_loss(targets, q_values, delta=10.0)

    raise ValueError('Entered unknown value for loss ' + loss_type)
