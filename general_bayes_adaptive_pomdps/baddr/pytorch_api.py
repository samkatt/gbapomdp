""" some pytorch integration things """

import torch

_DEVICE = "cpu"


def set_pytorch_seed(seed: int) -> None:
    """sets the random seed for pytorch

    Args:
         seed: (`int`):

    RETURNS (`None`):

    """
    torch.manual_seed(seed)  # type: ignore


def set_device(use_gpu: bool) -> None:
    """set what device to use (true for gpu)

    Args:
         use_gpu: (`bool`): whether to use the gpu

    RETURNS (`None`):

    """

    global _DEVICE
    _DEVICE = "cuda:0" if use_gpu else "cpu"


def device() -> str:
    """returns the current pytorch device

    RETURNS (`str`):

    """
    return _DEVICE
