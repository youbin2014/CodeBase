# base_dataset.py

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError("Please Implement the __len__ method")

    def __getitem__(self, idx):
        raise NotImplementedError("Please Implement the __getitem__ method")

    def get_train_dataset(self):
        raise NotImplementedError("Please Implement the get_train_dataset method")

    def get_test_dataset(self):
        raise NotImplementedError("Please Implement the get_test_dataset method")
