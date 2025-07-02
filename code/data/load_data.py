import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


def get_cifar10_dataloaders(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Train batches:", len(train_loader))
    print("Test batches:", len(test_loader))

    return train_loader, test_loader
