"""Basic utils for tensor handling."""

import torch


def atleast_kd(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Add dimensions to the tensor x until it reaches k dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    k : int
        Number of desired dimensions.

    Returns
    -------
    torch.Tensor
        The input tensor x but with k dimensions.
    """
    if k <= x.dim():
        msg = "The number of desired dimensions should be > x.dim()"
        raise ValueError(msg)
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)


def is_tensor(x: object) -> bool:
    """
    Check whether the input is a PyTorch tensor.

    Parameters
    ----------
    x : object
        Input object to check.

    Returns
    -------
    bool
        True if x is an instance of torch.Tensor, False otherwise.
    """
    return isinstance(x, torch.Tensor)


def is_list_of_tensors(x: object) -> bool:
    """
    Check whether the input is a list of PyTorch tensors.

    Parameters
    ----------
    x : object
        Input object to check.

    Returns
    -------
    bool
        True if x is a list and all its elements are instances
        of torch.Tensor, False otherwise.
    """
    if not isinstance(x, list) and not isinstance(x, tuple):
        return False

    return all(isinstance(elem, torch.Tensor) for elem in x)

def is_tuple_of_dict(y: object) -> bool:
    """
    Check whether the target is a tuple of dict.

    Parameters
    ----------
    x : object
        Input object to check.

    Returns
    -------
    bool
        True if x is a list and all its elements are instances
        of torch.Tensor, False otherwise.
    """
    if not isinstance(y, tuple):
        return False

    return all(isinstance(elem, dict) for elem in y)

