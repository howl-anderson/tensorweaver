import pickle
import os


def save(obj, f):
    """Mock implementation of torch.save to make it compatible with PyTorch-style saving.

    Args:
        obj: Object to save (can be model state_dict or any serializable object)
        f (str): Path where to save the object
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(f), exist_ok=True)

    # Save using pickle
    with open(f, "wb") as f:
        pickle.dump(obj, f)
