"""
Neural network model for MNIST/CIFAR-10 classification.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron for image classification."""

    def __init__(self, input_size: int, hidden_dims: list, num_classes: int = 10):
        super().__init__()
        layers = []
        in_dim = input_size
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
            ])
            in_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.classifier(x)


def get_model(dataset: str = "mnist", num_classes: int = 10) -> nn.Module:
    """Create model with appropriate input size for dataset."""
    if dataset == "mnist":
        input_size = 28 * 28  # 784
    elif dataset == "cifar10":
        input_size = 32 * 32 * 3  # 3072
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return MLP(
        input_size=input_size,
        hidden_dims=[512, 256, 128],
        num_classes=num_classes,
    )
