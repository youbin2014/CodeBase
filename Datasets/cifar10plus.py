import os
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
from .base_dataset import BaseDataset


class CIFAR10PlusDataset(BaseDataset):
    def __init__(self, root='./data', additional_data_root='/mnt/data/semantic_confusion_dataset/additional_CIFAR10_dataset', transform=None):
        super(CIFAR10PlusDataset, self).__init__()
        self.root = root
        self.additional_data_root = additional_data_root
        self.transform = transform if transform else transforms.ToTensor()
        self.cifar10_train = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True,
                                                     transform=self.transform)
        self.cifar10_classes = self.cifar10_train.classes

    def _relabel_additional_data(self):
        datasets = []

        # Update the transform to include resizing to 32x32
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize images to CIFAR10 resolution
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        additional_dataset = ImageFolder(root=self.additional_data_root,
                              transform=transform)


        # Concatenate all the datasets
        # combined_dataset = ConcatDataset(datasets)
        return additional_dataset

    def get_train_dataset(self):
        # Original CIFAR10 training data

        # Additional CIFAR10 training data with relabeled targets
        additional_data_train = self._relabel_additional_data()

        # Combining the two datasets
        combined_train = ConcatDataset([self.cifar10_train, additional_data_train])

        return combined_train

    def get_test_dataset(self):
        # Standard CIFAR10 test set
        return torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=transforms.ToTensor())
