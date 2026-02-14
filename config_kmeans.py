"""
Configuration for K-Means clustering - ML System Optimization Assignment
"""

# Dataset
DATASET = "mnist"  # mnist or blobs (synthetic)
DATA_DIR = "./data"
N_SAMPLES = 1500   # Digits has 1797; use subset for faster runs

# K-Means
N_CLUSTERS = 10
MAX_ITERS = 100

# Parallel
N_WORKERS = -1  # -1 = use all CPU cores (for parallel script)
