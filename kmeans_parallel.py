"""
Parallel K-Means: multiprocessing for assignment step.
Each worker processes a chunk of data points; centroids updated via reduction.
"""

import argparse
import os
import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import load_digits, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from config_kmeans import (
    DATASET,
    N_SAMPLES,
    N_CLUSTERS,
    MAX_ITERS,
    N_WORKERS,
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


def _assign_chunk(X_chunk: np.ndarray, centroids: np.ndarray):
    """
    Worker: assign chunk of points to nearest centroid.
    Returns (labels_chunk, sum_per_cluster, count_per_cluster).
    """
    k = centroids.shape[0]
    dists = np.sum((X_chunk[:, None] - centroids) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)

    sums = np.zeros_like(centroids)
    counts = np.zeros(k, dtype=np.int64)
    for j in range(k):
        mask = labels == j
        counts[j] = mask.sum()
        if counts[j] > 0:
            sums[j] = X_chunk[mask].sum(axis=0)
    return sums, counts


def kmeans_parallel(
    X: np.ndarray,
    k: int,
    max_iters: int,
    n_workers: int = -1,
    seed: int = 42,
):
    """
    Parallel K-Means: assignment step distributed across workers via joblib.
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

    n_chunks = os.cpu_count() if n_workers == -1 else n_workers
    n_chunks = max(1, min(n_chunks, n_samples))

    for _ in range(max_iters):
        chunks = np.array_split(X, n_chunks)
        results = Parallel(n_jobs=n_workers)(
            delayed(_assign_chunk)(chunk, centroids) for chunk in chunks
        )

        # Aggregate: sum and count per cluster
        total_sums = np.zeros_like(centroids)
        total_counts = np.zeros(k, dtype=np.int64)
        for sums, counts in results:
            total_sums += sums
            total_counts += counts

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            if total_counts[j] > 0:
                new_centroids[j] = total_sums[j] / total_counts[j]
            else:
                new_centroids[j] = centroids[j]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Final assignment (single pass or parallel) for labels
    dists = np.sum((X[:, None] - centroids) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    return centroids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--k", type=int, default=N_CLUSTERS)
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    print("Loading data...")
    X, y_true = load_data(args.dataset, args.n_samples)
    print(f"Data shape: {X.shape}")

    n_workers = args.n_workers if args.n_workers > 0 else -1
    print(f"\n--- Parallel K-Means ({n_workers} workers) ---\n")
    start = time.perf_counter()
    centroids, labels = kmeans_parallel(
        X, args.k, args.max_iters, n_workers=n_workers
    )
    elapsed = time.perf_counter() - start

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
