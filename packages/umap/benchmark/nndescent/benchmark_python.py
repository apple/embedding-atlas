#!/usr/bin/env python3
"""Benchmark for Python PyNNDescent.

Usage:
    python benchmark_python.py <data_dir> <metric> [--output results.csv]

Reads a data directory containing meta.json, data.bin, and optionally
truth_{metric}.bin. Builds PyNNDescent index on ALL data, then computes
recall against ground truth if available. Appends one CSV row to output.
"""

import json
import os
import sys
import time

import numpy as np


def read_meta(data_dir: str):
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    return meta["n_points"], meta["dim"], meta["k"]


def compute_recall(true_indices: np.ndarray, approx_indices: np.ndarray) -> float:
    n = true_indices.shape[0]
    k = true_indices.shape[1]
    total = 0
    for i in range(n):
        true_set = set(true_indices[i])
        approx_k = min(k, approx_indices.shape[1])
        for j in range(approx_k):
            if approx_indices[i, j] in true_set:
                total += 1
    return total / (n * k)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: benchmark_python.py <data_dir> <metric> [--output results.csv]",
            file=sys.stderr,
        )
        sys.exit(1)

    data_dir = sys.argv[1]
    metric = sys.argv[2]
    output_path = "bench/results.csv"
    if len(sys.argv) > 4 and sys.argv[3] == "--output":
        output_path = sys.argv[4]

    n, dim, k = read_meta(data_dir)
    print(
        f"Python benchmark: n={n}, dim={dim}, k={k}, metric={metric}", file=sys.stderr
    )

    # Load data
    print("  Loading data...", file=sys.stderr)
    data = np.fromfile(os.path.join(data_dir, "data.bin"), dtype=np.float32).reshape(
        n, dim
    )

    # Load ground truth
    truth_path = os.path.join(data_dir, f"truth_{metric}.bin")
    has_truth = os.path.exists(truth_path) and os.path.getsize(truth_path) > 0
    truth = None
    if has_truth:
        truth = np.fromfile(truth_path, dtype=np.int32).reshape(n, k)

    from pynndescent import NNDescent

    # Determine n_jobs from environment
    single_cpu = os.environ.get("BENCH_SINGLE_CPU", "0") == "1"
    n_jobs_val = 1 if single_cpu else -1
    if single_cpu:
        print("  Single-CPU mode: n_jobs=1", file=sys.stderr)

    # Warm up Numba JIT compilation on a tiny throwaway dataset
    print("  Warming up Numba JIT...", file=sys.stderr)
    warmup_data = np.random.RandomState(0).rand(100, dim).astype(np.float32)
    _ = NNDescent(
        warmup_data,
        metric=metric,
        n_neighbors=min(k, 10),
        random_state=0,
        n_jobs=n_jobs_val,
    )
    del warmup_data, _

    # Build index on ALL data
    # PyNNDescent includes self-loops (point as its own neighbor) at position 0,
    # which is consistent with our Rust implementation.
    print(f"  Building index on {n} points...", file=sys.stderr)
    t0 = time.perf_counter()
    nnd = NNDescent(
        data, metric=metric, n_neighbors=k, random_state=42, n_jobs=n_jobs_val
    )
    build_time = time.perf_counter() - t0
    print(f"  Build time: {build_time:.3f}s", file=sys.stderr)

    # Compute recall against ground truth
    if truth is not None:
        approx_indices = nnd.neighbor_graph[0].copy()
        # Ensure self-neighbor is at position 0 for each point, consistent
        # with our Rust post-processing.  PyNNDescent doesn't guarantee self
        # is present (it only enters via random init / iteration), so we
        # insert it explicitly to match the ground truth format.
        for i in range(n):
            row = approx_indices[i]
            positions = np.where(row == i)[0]
            if len(positions) > 0 and positions[0] == 0:
                continue  # already at position 0
            if len(positions) > 0:
                # self found at some later position; shift elements right
                pos = positions[0]
                approx_indices[i, 1 : pos + 1] = approx_indices[i, 0:pos]
            else:
                # self not found; shift everything right (drop last)
                approx_indices[i, 1:] = approx_indices[i, :-1]
            approx_indices[i, 0] = i
        recall = compute_recall(truth, approx_indices)
        print(f"  Recall: {recall:.4f}", file=sys.stderr)
        recall_str = f"{recall:.4f}"
    else:
        print("  No ground truth, skipping recall", file=sys.stderr)
        recall_str = "N/A"

    # Write CSV row
    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    with open(output_path, "a") as f:
        if write_header:
            f.write("implementation,n_points,dim,metric,build_time_s,recall\n")
        f.write(f"python,{n},{dim},{metric},{build_time:.3f},{recall_str}\n")

    print(f"  Result appended to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
