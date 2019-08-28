""" utility functions for networks """

from typing import List
import tensorflow as tf


def update_variables(
        targets: List[tf.Tensor], updates: List[tf.Tensor]) -> tf.Operation:
    """ updates `targets` to have values of `updates`

    Assumes the elements of `targets` and `updates` are of the same type

    Args:
         targets: (`List`): list of tensorflow tensors
         updates: (`List`): list of tensorflow tensors

    """

    with tf.name_scope('update_variables'):
        update_target_op = []
        for target, update in zip(sorted(targets, key=lambda v: v.name),
                                  sorted(updates, key=lambda v: v.name)):
            update_target_op.append(target.assign(update))

    return tf.group(*update_target_op)
