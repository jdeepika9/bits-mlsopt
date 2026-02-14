"""
Configuration for ML System Optimization Assignment - Distributed Neural Network Training
"""

# Dataset
DATASET = "mnist"  # mnist or cifar10
DATA_DIR = "./data"

# Model
HIDDEN_DIMS = [512, 256, 128]
NUM_CLASSES = 10

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Device
DEVICE = "cuda"  # cuda or cpu
