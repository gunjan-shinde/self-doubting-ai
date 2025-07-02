from torchvision import datasets, transforms
from torch.utils.data import random_split


def get_train_val_datasets(val_split=0.2):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Example dataset: MNIST
    full_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, val_dataset
