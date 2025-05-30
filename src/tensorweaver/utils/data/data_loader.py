# simple_dataloader.py
import random
from tensorweaver.autodiff.tensor import Tensor
import numpy as np
from typing import Iterable, Callable, List, Any


def default_collate(batch: List[Any]):
    """Aggregates samples in a batch element-wise:
    [(x1, y1), (x2, y2)] -> ( [x1, x2], [y1, y2] )"""
    elem = batch[0]
    if isinstance(elem, (tuple, list)):
        transposed = zip(*batch)  # Reorganize by position
        return [default_collate(samples) for samples in transposed]
    if isinstance(elem, (np.ndarray, float, int)):
        return Tensor(np.array(batch))
    if isinstance(elem, Tensor):
        return Tensor(np.stack([x.data for x in batch]))
    return batch  # Return other types directly


class DataLoader(Iterable):
    """Minimal DataLoader: single process + random shuffle + auto batching"""

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Callable = default_collate,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __iter__(self):
        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idxs)

        # Yields an index slice for a batch each time
        batch, bs = [], self.batch_size
        for idx in idxs:
            batch.append(self.dataset[idx])
            if len(batch) == bs:
                yield self.collate_fn(batch)
                batch = []

        # Handle the remainder if it's less than batch_size
        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self):
        """Returns the number of batches: for tqdm progress bar"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
