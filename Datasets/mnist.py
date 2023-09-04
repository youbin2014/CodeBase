# mnist.py

import torchvision
import torchvision.transforms as transforms
from .base_dataset import BaseDataset

class MNISTDataset(BaseDataset):
    def __init__(self, root='./data', transform=None):
        super(MNISTDataset, self).__init__()
        self.root = root
        self.transform = transform if transform else transforms.ToTensor()

    def get_train_dataset(self):
        return torchvision.datasets.MNIST(root=self.root, train=True, download=True, transform=self.transform)

    def get_test_dataset(self):
        return torchvision.datasets.MNIST(root=self.root, train=False, download=True, transform=transforms.ToTensor())
