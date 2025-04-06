from numpy.typing import NDArray

import numpy as np


def as_ndarray(x, like_to : NDArray = None) -> NDArray:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if like_to is not None:
        x = np.asarray(x, dtype=like_to.dtype)

    return x


def as_variable(x, like_to: "Variable" = None) -> "Variable":
    # lazy load to avoid import circle
    from tensorweaver.autodiff.variable import Variable

    if not isinstance(x, Variable):
        return Variable(as_ndarray(x, like_to.data if like_to else None))

    return x
