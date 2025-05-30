from tensorweaver.utils.data.dataset import Dataset


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]
