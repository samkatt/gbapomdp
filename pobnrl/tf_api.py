""" interactions with tensorflow """

from contextlib import contextmanager
from typing import Any
import os
import tensorflow as tf

from misc import POBNRLogger

tf.logging.set_verbosity(tf.logging.ERROR)

# please, for the love of everything good in this world, don't refer to this
_SESS: tf.Session = None
_TF_BOARD_WRITER: tf.summary.FileWriter = None
_NUM_OPERATIONS: int = 0


@contextmanager
def tf_session(use_gpu: bool, tensorboard_name: str):
    """ used as context to run TF in

    e.g.:
    with tf_session(False, ""):

        ...
        tf_run(...)


    Args:
         use_gpu: (`bool`): whether to use gpus
         tensorboard_name: (`str`) subdirectory of tensorboard to write to
    """

    # __enter__
    global _SESS                # pylint: disable=global-statement
    global _TF_BOARD_WRITER     # pylint: disable=global-statement
    global _NUM_OPERATIONS      # pylint: disable=global-statement

    assert _SESS is None, "Please initiate tf_wrapper only once"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # avoid print statements

    logger = POBNRLogger('tf_session')
    logger.log(POBNRLogger.LogLevel.V1, "initiating tensorflow session")

    tf_config = tf.ConfigProto(
        device_count={'GPU': int(use_gpu)},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )

    _SESS = tf.Session(config=tf_config)
    if tensorboard_name:
        _TF_BOARD_WRITER = tf.summary.FileWriter(f'.tensorboard/{tensorboard_name}', _SESS.graph)
    _NUM_OPERATIONS = 0

    yield _SESS

    # __exit__()

    logger.log(POBNRLogger.LogLevel.V1, "closing tensorflow session")

    _SESS.close()
    _SESS = None

    if tensorboard_name:
        _TF_BOARD_WRITER.close()
        _TF_BOARD_WRITER = None


def tf_run(operations, **kwargs) -> Any:
    """ runs a tf session """
    global _NUM_OPERATIONS  # pylint: disable=global-statement
    _NUM_OPERATIONS += 1
    return _SESS.run(operations, **kwargs)


def tf_board_write(summary: tf.summary.Summary) -> None:
    """  writes a summary to file for tensorboard

    Args:
         summary: (`tf.summary.Summary`): the thing to write away to tensorboard

    RETURNS (`None`):

    """
    if _TF_BOARD_WRITER:
        _TF_BOARD_WRITER.add_summary(summary, _NUM_OPERATIONS)
