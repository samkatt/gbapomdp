""" defines loss functions """

import tensorflow as tf


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

    Given some input, a Q-net can estimate the Q-values and some target q-values.
    This function returns the loss over the estimated q-values, given the type of loss

    Args:
         q_values: q-value estimates
         targets: target q-value estimates
         loss_type: (`str`): is "rmse" or "huber" to indicate type of loss to use

    """

    # training operation loss
    if loss_type == "rmse":
        return tf.losses.mean_squared_error(targets, q_values)
    if loss_type == "huber":
        return tf.losses.huber_loss(targets, q_values, delta=10.0)

    raise ValueError('Entered unknown value for loss ' + loss_type)
