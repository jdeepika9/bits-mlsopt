"""
Baseline: Single-GPU/single-process neural network training.
Use this for correctness comparison and speedup benchmarking.
"""

import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    DATASET,
    DATA_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)
from model import get_model
from dataset import get_dataloaders


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
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

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on test set."""
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

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    model = get_model(dataset=args.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Metrics for assignment report
    total_samples = 0
    start_time = time.perf_counter()

    print("\n--- Baseline (Single GPU) Training ---\n")
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        epoch_time = time.perf_counter() - epoch_start
        total_samples += len(train_loader.dataset)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.perf_counter() - start_time
    throughput = total_samples / total_time

    print("\n--- Baseline Summary ---")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
