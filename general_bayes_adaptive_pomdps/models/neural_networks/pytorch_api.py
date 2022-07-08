""" some pytorch integration things """

import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def set_pytorch_seed(seed: int) -> None:
    """sets the random seed for pytorch

    Args:
         seed: (`int`):

    RETURNS (`None`):

    """
    torch.manual_seed(seed)  # type: ignore


def device() -> str:
    """returns the current pytorch device

    RETURNS (`str`):

    """
    return DEVICE
