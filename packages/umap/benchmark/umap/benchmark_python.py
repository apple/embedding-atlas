#!/usr/bin/env python3
"""Python UMAP benchmark using umap-learn.

Usage:
    uv run --with umap-learn,numpy bench/benchmark_python.py <data_dir> <metric> [--output results.csv]

Reads meta.json + data.bin, runs UMAP with matching parameters, saves embedding and timing.
"""

import argparse
import json
import os
import time

import numpy as np


def read_f32_bin(path, n, dim):
    """Read a binary file of float32 little-endian values."""
    data = np.fromfile(path, dtype="<f4")
    return data.reshape(n, dim)


def write_f32_bin(path, data):
    """Write a numpy array as float32 little-endian."""
    data.astype("<f4").tofile(path)


def main():
    parser = argparse.ArgumentParser(description="Python UMAP benchmark")
    parser.add_argument("data_dir", help="Directory with meta.json and data.bin")
    parser.add_argument("metric", help="Distance metric (euclidean or cosine)")
    parser.add_argument("--output", default=None, help="CSV output path")
    args = parser.parse_args()

    data_dir = args.data_dir
    metric = args.metric
    output_path = args.output or os.path.join(data_dir, "results.csv")

    # Read metadata
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_points"]
    dim = meta["dim"]

    print(f"Python UMAP benchmark: n={n}, dim={dim}, metric={metric}", flush=True)

    # Load data
    print("  Loading data...", flush=True)
    data = read_f32_bin(os.path.join(data_dir, "data.bin"), n, dim)

    # Run UMAP
    import umap as umap_lib

    print("  Running UMAP...", flush=True)
    single_cpu = os.environ.get("BENCH_SINGLE_CPU", "0") == "1"
    n_jobs_val = 1 if single_cpu else -1
    if single_cpu:
        print("  Single-CPU mode: n_jobs=1", flush=True)

    reducer = umap_lib.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric=metric,
        n_jobs=n_jobs_val,
    )

    t0 = time.perf_counter()
    embedding = reducer.fit_transform(data)
    elapsed = time.perf_counter() - t0

    print(f"  UMAP time: {elapsed:.3f}s", flush=True)

    # Save embedding
    emb_path = os.path.join(data_dir, f"python_{metric}_embedding.bin")
    write_f32_bin(emb_path, embedding.astype(np.float32))
    print(f"  Saved embedding to {emb_path}", flush=True)

    # Append to results CSV
    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    with open(output_path, "a") as f:
        if write_header:
            f.write("implementation,n_points,dim,metric,time_s\n")
        f.write(f"python,{n},{dim},{metric},{elapsed:.3f}\n")

    print(f"  Result appended to {output_path}", flush=True)


if __name__ == "__main__":
    main()
