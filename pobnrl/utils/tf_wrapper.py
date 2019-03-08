""" wrapper functions for interaction with tensorflow

Simply has a 'init' and 'getter' function to initiate and return sessions

"""

import tensorflow as tf

# please, for the love of everything good in this world, don't refer to
# this directly
____sess = None


def init():
    """init initiates the wrapper (called once)

    anything done with TF before calling this function is bogus
    """

    global ____sess
    assert ____sess is None, "Please initiate tf_wrapper only once"

    tf.reset_default_graph()

    tf_config = tf.ConfigProto(
        device_count={'GPU': 0},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
    )

    ____sess = tf.Session(config=tf_config)


def get_session():
    """ returns current session """

    global ____sess

    if ____sess is None:
        init()

    return ____sess
