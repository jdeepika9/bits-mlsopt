"""
Run baseline and DDP training for benchmarking.
Produces metrics for assignment report: speedup, throughput, accuracy.
"""

import subprocess
import sys
import time


def run_cmd(cmd, label):
    """Run a command and return elapsed time."""
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    start = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"Error: {label} failed with code {result.returncode}")
        return None
    return elapsed


def main():
    epochs = 5  # Use fewer epochs for quick benchmark
    batch_size = 128

    # 1. Baseline (single GPU)
    baseline_time = run_cmd(
        [sys.executable, "train_baseline.py", "--epochs", str(epochs), "--batch-size", str(batch_size)],
        "Baseline (Single GPU)"
    )

    # 2. DDP with 2 GPUs (if available)
    ddp_time = run_cmd(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "train_ddp.py",
         "--epochs", str(epochs), "--batch-size", str(batch_size)],
        "DDP (2 GPUs)"
    )

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    if baseline_time:
        print(f"Baseline time: {baseline_time:.2f}s")
    if ddp_time:
        print(f"DDP (2 GPUs) time: {ddp_time:.2f}s")
        if baseline_time and baseline_time > 0:
            speedup = baseline_time / ddp_time
            print(f"Speedup: {speedup:.2f}x")
            print(f"Scaling efficiency: {speedup/2*100:.1f}%")


if __name__ == "__main__":
    main()
