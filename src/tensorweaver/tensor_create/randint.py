import numpy as np
from tensorweaver.autodiff.tensor import Tensor


def randint(low=0, high=None, size=None, dtype=None):
    if dtype is None:
        dtype = np.int64

    if high is None:
        high = low
        low = 0
    if size is None:
        size = ()

    # Type checking
    if not isinstance(low, (int, np.integer)):
        raise TypeError(f"low must be an integer, not {type(low)}")
    if not isinstance(high, (int, np.integer)):
        raise TypeError(f"high must be an integer, not {type(high)}")

    # Value checking
    if high <= low:
        raise ValueError(f"high ({high}) must be greater than low ({low})")
    if isinstance(size, (list, tuple)):
        if any(s < 0 for s in size):
            raise ValueError(f"size cannot contain negative values")
    elif isinstance(size, int) and size < 0:
        raise ValueError(f"size cannot be negative")

    return Tensor(np.random.randint(low, high, size=size, dtype=dtype))
