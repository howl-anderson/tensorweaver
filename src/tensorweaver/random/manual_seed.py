import numpy as np


def manual_seed(seed: int) -> None:
    """
    Set the seed for the random number generator.
    """
    np.random.seed(seed)
