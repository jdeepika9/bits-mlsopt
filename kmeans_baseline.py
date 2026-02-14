"""
Baseline: Single-process K-Means clustering.
Use for correctness comparison and speedup benchmarking.
"""

import argparse
import time
import numpy as np
from sklearn.datasets import load_digits, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from config_kmeans import (
    DATASET,
    N_SAMPLES,
    N_CLUSTERS,
    MAX_ITERS,
)


def load_data(dataset: str, n_samples: int = None):
    """Load dataset for clustering. Returns (X, y) where y is true labels (for ARI)."""
    if dataset == "mnist":
        data = load_digits()
        X, y = data.data, data.target
    elif dataset == "blobs":
        X, y = make_blobs(n_samples=10000, n_features=64, centers=10, random_state=42)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if n_samples and n_samples < len(X):
        rng = np.random.default_rng(42)
        n_samples = min(n_samples, len(X))
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)
    return X, y


def kmeans_baseline(X: np.ndarray, k: int, max_iters: int, seed: int = 42):
    """
    Standard K-Means: single process, sequential assignment and update.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    # Initialize centroids: k-means++
    centroids = np.zeros((k, n_features))
    centroids[0] = X[rng.integers(n_samples)]
    for i in range(1, k):
        dist_sq = np.min(np.sum((X[:, None] - centroids[:i]) ** 2, axis=2), axis=1)
        probs = dist_sq / dist_sq.sum()
        centroids[i] = X[rng.choice(n_samples, p=probs)]

    for _ in range(max_iters):
        # Assignment: each point -> nearest centroid
        dists = np.sum((X[:, None] - centroids) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        # Update: new centroid = mean of assigned points
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--k", type=int, default=N_CLUSTERS)
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    args = parser.parse_args()

    print("Loading data...")
    X, y_true = load_data(args.dataset, args.n_samples)
    print(f"Data shape: {X.shape}")

    print("\n--- Baseline (Single Process) K-Means ---\n")
    start = time.perf_counter()
    centroids, labels = kmeans_baseline(X, args.k, args.max_iters)
    elapsed = time.perf_counter() - start

    # Metrics: inertia, silhouette, ARI (when ground truth available)
    inertia = sum(np.sum((X[labels == j] - centroids[j]) ** 2) for j in range(args.k))
    silhouette = silhouette_score(X, labels)
    ari = adjusted_rand_score(y_true, labels) if y_true is not None else None

    print(f"Converged in {len(np.unique(labels))} clusters")
    print(f"Training time: {elapsed:.2f}s")
    print(f"Inertia: {inertia:.2f}")
    print(f"Silhouette score: {silhouette:.4f}")
    if ari is not None:
        print(f"Adjusted Rand Index: {ari:.4f}")


if __name__ == "__main__":
    main()
