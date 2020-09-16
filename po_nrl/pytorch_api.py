""" some pytorch integration things """

from typing import Union, Optional
import numpy as np
import torch.utils.tensorboard

_DEVICE = 'cpu'

_TENSORBOARD_WRITER: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None


def set_random_seed(seed: int) -> None:
    """ sets the random seed for pytorch

    Args:
         seed: (`int`):

    RETURNS (`None`):

    """
    torch.manual_seed(seed)  # type: ignore


def set_tensorboard_logging(log_dir: str) -> None:
    """ set what directory to write tensorboard results to

    Args:
         log_dir: (`str`): what, if any, directory to write tensorboard results to

    RETURNS (`None`):

    """

    global _TENSORBOARD_WRITER  # pylint: disable=global-statement
    _TENSORBOARD_WRITER = torch.utils.tensorboard.SummaryWriter(  # type: ignore
        log_dir=f'.tensorboard/{log_dir}'
    )


def set_device(use_gpu: bool) -> None:
    """ set what device to use (true for gpu)

    Args:
         use_gpu: (`bool`): whether to use the gpu

    RETURNS (`None`):

    """

    global _DEVICE  # pylint: disable=global-statement
    _DEVICE = 'cuda:0' if use_gpu else 'cpu'


def device() -> str:
    """ returns the current pytorch device

    RETURNS (`str`):

    """
    return _DEVICE


def log_tensorboard(tag: str, val: Union[float, np.ndarray], step: int) -> None:
    """ logs a scalar to tensorboard

    Args:
         tag: (`str`): the 'topic' to write results to
         val: (`Union[float, np.ndarray]`): either a scalar or a histogram
         step: (`int`): where on the x-axis the result should be written to

    """

    assert tensorboard_logging(), 'please first verify logging is on'
    assert _TENSORBOARD_WRITER

    if np.isscalar(val):
        _TENSORBOARD_WRITER.add_scalar(tag, val, step)  # type: ignore
    else:
        _TENSORBOARD_WRITER.add_histogram(tag, val, step)  # type: ignore


def tensorboard_logging() -> bool:
    """ returns whether we are logging to tensorboard

    Args:

    RETURNS (`bool`):

    """
    return _TENSORBOARD_WRITER is not None
