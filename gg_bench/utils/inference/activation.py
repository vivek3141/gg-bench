import numpy as np
from numpy.typing import ArrayLike


def softmax(x: ArrayLike) -> np.ndarray:
    """
    Compute the softmax of a vector or matrix.

    The softmax function is defined as:
    softmax(x)_i = exp(x_i) / sum(exp(x_j))

    This implementation is numerically stable by subtracting the maximum
    value from each input before exponentiating.

    Args:
        x (ArrayLike): Input array. Can be a vector or matrix.

    Returns:
        np.ndarray: Softmax of the input array. Same shape as input.

    Raises:
        ValueError: If input is not a 1D or 2D array.

    Example:
        >>> softmax(np.array([1.0, 2.0, 3.0]))
        array([0.09003057, 0.24472847, 0.66524096])
    """
    x_array = np.asarray(x)
    if x_array.ndim > 2:
        raise ValueError("Input must be a 1D or 2D array.")
    exp_x = np.exp(x_array - np.max(x_array, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
