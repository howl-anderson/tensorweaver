import pytest
from tensorweaver.utils.data.dataset import Dataset
from tensorweaver.utils.data.tensor_dataset import TensorDataset
from tensorweaver.utils.data.data_loader import DataLoader
from tensorweaver.autodiff.tensor import Tensor
import numpy as np  # Assuming tensors are numpy arrays for testing


class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def test_dataset_getitem():
    mock_data = [1, 2, 3]
    dataset = MockDataset(mock_data)
    assert dataset[0] == 1
    assert dataset[1] == 2
    assert dataset[2] == 3


def test_dataset_add_not_implemented():
    dataset1 = MockDataset([1, 2])
    dataset2 = MockDataset([3, 4])
    with pytest.raises(NotImplementedError):
        _ = dataset1 + dataset2


def test_tensor_dataset_getitem_and_len():
    t1 = np.array([[1, 2], [3, 4], [5, 6]])
    t2 = np.array([[7, 8], [9, 10], [11, 12]])
    tensor_dataset = TensorDataset(t1, t2)

    assert len(tensor_dataset) == 3
    item0 = tensor_dataset[0]
    assert isinstance(item0, tuple)
    assert np.array_equal(item0[0], np.array([1, 2]))
    assert np.array_equal(item0[1], np.array([7, 8]))

    item1 = tensor_dataset[1]
    assert isinstance(item1, tuple)
    assert np.array_equal(item1[0], np.array([3, 4]))
    assert np.array_equal(item1[1], np.array([9, 10]))


def test_dataloader_iteration_batch_size_1():
    data = [1, 2, 3, 4, 5]
    dataset = MockDataset(data)
    dataloader = DataLoader(dataset, batch_size=1)
    batches = list(dataloader)
    assert len(batches) == 5
    assert isinstance(batches[0], Tensor)
    assert np.array_equal(batches[0].data, np.array([1]))
    assert np.array_equal(batches[1].data, np.array([2]))
    assert np.array_equal(batches[2].data, np.array([3]))
    assert np.array_equal(batches[3].data, np.array([4]))
    assert np.array_equal(batches[4].data, np.array([5]))


def test_dataloader_iteration_batch_size_2():
    data = [1, 2, 3, 4, 5]
    dataset = MockDataset(data)
    dataloader = DataLoader(dataset, batch_size=2)
    batches = list(dataloader)
    assert len(batches) == 3
    assert isinstance(batches[0], Tensor)
    assert np.array_equal(batches[0].data, np.array([1, 2]))
    assert np.array_equal(batches[1].data, np.array([3, 4]))
    assert np.array_equal(batches[2].data, np.array([5]))


def test_dataloader_drop_last():
    data = [1, 2, 3, 4, 5]
    dataset = MockDataset(data)
    dataloader = DataLoader(dataset, batch_size=2, drop_last=True)
    batches = list(dataloader)
    assert len(batches) == 2
    assert isinstance(batches[0], Tensor)
    assert np.array_equal(batches[0].data, np.array([1, 2]))
    assert np.array_equal(batches[1].data, np.array([3, 4]))
    assert len(dataloader) == 2


def test_dataloader_shuffle():
    # Since shuffle is random, we mainly test if the number of batches and the size of each batch are correct
    # and if all elements have appeared without duplicates (unless the original data has duplicates)
    data = list(range(10))
    dataset = MockDataset(data)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    all_yielded_items = []
    for batch in dataloader:
        assert isinstance(batch, Tensor)
        assert len(batch.data) <= 3
        all_yielded_items.extend(batch.data.tolist())

    assert len(all_yielded_items) == 10
    assert sorted(all_yielded_items) == data  # Ensure all elements are iterated through


def test_dataloader_collate_fn():
    # Use TensorDataset to test the default collate_fn
    t1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    t2 = np.array([[9, 10], [11, 12], [13, 14], [15, 16]])
    tensor_dataset = TensorDataset(t1, t2)
    dataloader = DataLoader(tensor_dataset, batch_size=2)

    batches = list(dataloader)
    assert len(batches) == 2

    batch1 = batches[0]
    assert isinstance(batch1, list)
    assert len(batch1) == 2  # dataset returns a tuple, after collate it should be a list of tensors
    assert isinstance(batch1[0], Tensor)
    assert isinstance(batch1[1], Tensor)
    assert np.array_equal(batch1[0].data, np.array([[1, 2], [3, 4]]))
    assert np.array_equal(batch1[1].data, np.array([[9, 10], [11, 12]]))

    batch2 = batches[1]
    assert isinstance(batch2, list)
    assert len(batch2) == 2
    assert isinstance(batch2[0], Tensor)
    assert isinstance(batch2[1], Tensor)
    assert np.array_equal(batch2[0].data, np.array([[5, 6], [7, 8]]))
    assert np.array_equal(batch2[1].data, np.array([[13, 14], [15, 16]]))


def test_dataloader_len():
    dataset = MockDataset(list(range(10)))
    dataloader1 = DataLoader(dataset, batch_size=3, drop_last=False)
    assert len(dataloader1) == 4  # (10 + 3 - 1) // 3 = 4

    dataloader2 = DataLoader(dataset, batch_size=3, drop_last=True)
    assert len(dataloader2) == 3  # 10 // 3 = 3

    dataloader3 = DataLoader(dataset, batch_size=10, drop_last=False)
    assert len(dataloader3) == 1

    dataloader4 = DataLoader(dataset, batch_size=10, drop_last=True)
    assert len(dataloader4) == 1

    dataset_empty = MockDataset([])
    dataloader5 = DataLoader(dataset_empty, batch_size=3, drop_last=False)
    assert len(dataloader5) == 0

    dataloader6 = DataLoader(dataset_empty, batch_size=3, drop_last=True)
    assert len(dataloader6) == 0
