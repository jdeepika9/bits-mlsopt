"""
Distributed Data Parallel (DDP) training across multiple GPUs.
Use torchrun to launch: torchrun --nproc_per_node=N train_ddp.py
"""

import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from config import (
    DATASET,
    DATA_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)
from model import get_model


def setup_ddp():
    """Initialize distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def get_ddp_dataloaders(dataset_name, data_dir, batch_size, world_size, rank):
    """Create dataloaders with DistributedSampler for DDP."""
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ])
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader, train_sampler


def train_epoch(model, loader, criterion, optimizer, device, epoch, train_sampler):
    """Train for one epoch with DDP."""
    model.train()
    if train_sampler:
        train_sampler.set_epoch(epoch)

    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    # Aggregate metrics across all ranks
    metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, correct, total = metrics[0].item(), metrics[1].item(), metrics[2].item()
    total_loss /= (len(loader) * dist.get_world_size())  # avg over processes
    return total_loss, correct / total


def evaluate(model, loader, criterion, device, world_size):
    """Evaluate on test set (rank 0 only for simplicity, or all-reduce)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, correct, total = metrics[0].item(), metrics[1].item(), metrics[2].item()
    return total_loss / (len(loader) * world_size), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n--- DDP Training: {world_size} GPU(s) ---\n")

    train_loader, test_loader, train_sampler = get_ddp_dataloaders(
        args.dataset, args.data_dir, args.batch_size, world_size, rank
    )

    model = get_model(dataset=args.dataset).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_samples = len(train_loader.dataset) * world_size
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, train_sampler
        )
        epoch_time = time.perf_counter() - epoch_start
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, world_size)

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

    total_time = time.perf_counter() - start_time
    throughput = total_samples * args.epochs / total_time

    if rank == 0:
        print("\n--- DDP Summary ---")
        print(f"World size: {world_size} GPU(s)")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.1f} samples/sec")
        print(f"Final test accuracy: {test_acc:.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
