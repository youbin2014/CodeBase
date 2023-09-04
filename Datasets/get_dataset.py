import torch
from torchvision import transforms
from Datasets.cifar10 import CIFAR10Dataset
from Datasets.mnist import MNISTDataset
from Datasets.cifar10plus import CIFAR10PlusDataset
from typing import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float],device):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


class NormalizeLayer_MNIST(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float],device):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer_MNIST, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        return (input - self.means) / self.sds
def get_dataset(dataset_name,batch_size=64, num_workers=16, device=None):
    if dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = CIFAR10Dataset(transform=transform)
        num_classes = 10

        _CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
        _CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

    elif dataset_name.lower() == "cifar10plus":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = CIFAR10PlusDataset(transform=transform)
        num_classes = 10

        _CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
        _CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

    elif dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = MNISTDataset(transform=transform)
        num_classes = 10
        _MNIST_MEAN = [0.1307]
        _MNIST_STDDEV = [0.3081]

    elif dataset_name.lower() == "imagenet":

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        # dataset = IMAGENETDataset(transform=transform)
        num_classes= 1000
        _IMAGENET_MEAN = [0.485, 0.456, 0.406]
        _IMAGENET_STDDEV = [0.229, 0.224, 0.225]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(dataset.get_train_dataset(), batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset.get_test_dataset(), batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True)

    def get_normalize_layer(dataset: str) -> torch.nn.Module:
        """Return the dataset's normalization layer"""
        if dataset == "imagenet":
            return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV,device)
        elif dataset == "cifar10":
            return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV,device)
        elif dataset == "cifar10plus":
            return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV,device)
        elif dataset == "mnist":
            return NormalizeLayer_MNIST(_MNIST_MEAN, _MNIST_STDDEV,device)



    return train_loader, test_loader, num_classes,get_normalize_layer(dataset_name)
