from numpy.typing import NDArray

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorweaver.autodiff.tensor import Tensor


def as_ndarray(x, like_to: NDArray = None) -> NDArray:
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if like_to is not None:
        x = np.asarray(x, dtype=like_to.dtype)

    return x


def as_tensor(x, like_to: "Tensor" = None) -> "Tensor":
    # lazy load to avoid import circle
    from tensorweaver.autodiff.tensor import Tensor

    if not isinstance(x, Tensor):
        return Tensor(as_ndarray(x, like_to.data if like_to else None))

    return x
