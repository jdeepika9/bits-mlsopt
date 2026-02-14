"""
Dataset loading utilities for MNIST and CIFAR-10.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_transforms():
    """Transforms for MNIST."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_cifar10_transforms():
    """Transforms for CIFAR-10."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])


def get_dataloaders(
    dataset: str = "mnist",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
):
    """Create train and test dataloaders."""
    if dataset == "mnist":
        transform = get_mnist_transforms()
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
    elif dataset == "cifar10":
        transform = get_cifar10_transforms()
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
