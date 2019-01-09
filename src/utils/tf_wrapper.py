""" wrapper functions for interaction with tensorflow """

import tensorflow as tf

# please, for the love of everything good in this world, don't refer to this directly
____sess = None

def init():
    """ initiates this wrapper

    anything done with TF before calling this function is bogus
    """
    global ____sess
    assert ____sess is None, "Please initiate tf_wrapper only once"

    tf.reset_default_graph()

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        )

    ____sess = tf.Session(config=tf_config)


def get_session():
    """ returns the TF session """

    global ____sess

    assert ____sess is not None, "Please initiate tf_wrapper"

    return ____sess
