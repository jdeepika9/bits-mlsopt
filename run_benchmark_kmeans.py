"""
Run baseline and parallel K-Means for benchmarking.
Captures: time, inertia, silhouette score, ARI, speedup.
Outputs results for report and PDF generation.
"""

import json
import os
import sys
import time

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score

from kmeans_baseline import kmeans_baseline, load_data
from kmeans_parallel import kmeans_parallel


def run_benchmark(n_samples=10000, k=10, max_iters=50, dataset="mnist", n_workers=-1):
    """Run both baseline and parallel, return metrics dict."""
    print("Loading data...")
    X, y_true = load_data(dataset, n_samples)
    print(f"Data shape: {X.shape}, Dataset: {dataset}\n")

    results = {"dataset": dataset, "n_samples": n_samples, "k": k, "max_iters": max_iters}

    # Baseline
    print("--- Baseline (Single Process) ---")
    start = time.perf_counter()
    centroids_b, labels_b = kmeans_baseline(X, k, max_iters)
    t_baseline = time.perf_counter() - start

    inertia_b = sum(np.sum((X[labels_b == j] - centroids_b[j]) ** 2) for j in range(k))
    sil_b = silhouette_score(X, labels_b)
    ari_b = adjusted_rand_score(y_true, labels_b)

    results["baseline"] = {
        "time_s": round(t_baseline, 2),
        "inertia": round(float(inertia_b), 2),
        "silhouette": round(float(sil_b), 4),
        "ari": round(float(ari_b), 4),
    }
    print(f"Time: {t_baseline:.2f}s | Inertia: {inertia_b:.2f} | Silhouette: {sil_b:.4f} | ARI: {ari_b:.4f}\n")

    # Parallel
    n_w = n_workers if n_workers > 0 else os.cpu_count()
    print(f"--- Parallel ({n_w} workers) ---")
    start = time.perf_counter()
    centroids_p, labels_p = kmeans_parallel(X, k, max_iters, n_workers=n_workers)
    t_parallel = time.perf_counter() - start

    inertia_p = sum(np.sum((X[labels_p == j] - centroids_p[j]) ** 2) for j in range(k))
    sil_p = silhouette_score(X, labels_p)
    ari_p = adjusted_rand_score(y_true, labels_p)

    results["parallel"] = {
        "time_s": round(t_parallel, 2),
        "inertia": round(float(inertia_p), 2),
        "silhouette": round(float(sil_p), 4),
        "ari": round(float(ari_p), 4),
    }
    results["speedup"] = round(t_baseline / t_parallel, 2) if t_parallel > 0 else 0

    print(f"Time: {t_parallel:.2f}s | Inertia: {inertia_p:.2f} | Silhouette: {sil_p:.4f} | ARI: {ari_p:.4f}\n")

    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Baseline time: {t_baseline:.2f}s")
    print(f"Parallel time: {t_parallel:.2f}s")
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"Correctness: Inertia diff={abs(inertia_b-inertia_p):.2f}, ARI diff={abs(ari_b-ari_p):.4f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    results = run_benchmark(
        n_samples=args.n_samples,
        k=args.k,
        max_iters=args.max_iters,
        dataset=args.dataset,
        n_workers=args.n_workers,
    )
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
