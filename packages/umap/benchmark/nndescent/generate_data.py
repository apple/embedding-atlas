#!/usr/bin/env python3
"""Generate one benchmark dataset with ground truth.

Usage:
    python generate_data.py --size 10000 --dim 100 --k 15 [--seed 42]
                            [--truth-metrics euclidean cosine manhattan]
                            [--output-dir bench/data]

Generates: <output_dir>/<size>_<dim>/
  - meta.json:  {"n_points": N, "dim": D, "k": K, "dtype": "float32"}
  - data.bin:   raw float32 little-endian, row-major (n_points, dim)
  - truth_<metric>.bin: raw int32 (n_points, k) — brute-force NN indices
"""

import argparse
import json
import os
import time

import numpy as np

MAX_GROUND_TRUTH_POINTS = 100_000


def exact_knn(data: np.ndarray, k: int, metric: str) -> np.ndarray:
    """Compute exact k-NN using sklearn, including self-matches (consistent with pynndescent)."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    data64 = data.astype(np.float64)

    if metric == "cosine":
        # For cosine, L2-normalize then use euclidean: ||a-b||^2 = 2(1 - cos(a,b))
        # This gives identical rankings and avoids overflow in sklearn's cosine matmul.
        data64 = normalize(data64, norm="l2")
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    else:
        nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")

    nn.fit(data64)
    _, indices = nn.kneighbors(data64)
    # First column is self-match (distance 0); keep it for consistency with pynndescent
    return indices.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Generate one benchmark dataset")
    parser.add_argument("--size", type=int, required=True, help="Number of data points")
    parser.add_argument("--dim", type=int, required=True, help="Dimensionality")
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--truth-metrics",
        nargs="*",
        default=[],
        help="Metrics to compute brute-force ground truth for",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: bench/data)"
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data"
        )

    dirname = f"{args.size}_{args.dim}"
    dirpath = os.path.join(args.output_dir, dirname)
    os.makedirs(dirpath, exist_ok=True)

    meta_path = os.path.join(dirpath, "meta.json")
    data_path = os.path.join(dirpath, "data.bin")

    # Check if already generated with matching params
    if os.path.exists(meta_path) and os.path.exists(data_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if (
            meta.get("n_points") == args.size
            and meta.get("dim") == args.dim
            and meta.get("k") == args.k
        ):
            # Check if all requested truth files exist (not needed for large datasets)
            all_truth_exist = args.size > MAX_GROUND_TRUTH_POINTS or all(
                os.path.exists(os.path.join(dirpath, f"truth_{m}.bin"))
                and os.path.getsize(os.path.join(dirpath, f"truth_{m}.bin")) > 0
                for m in args.truth_metrics
            )
            if all_truth_exist:
                print(f"{dirname}: already exists, skipping")
                return

    # Generate data
    print(f"{dirname}: generating {args.size} x {args.dim} ...", end="", flush=True)
    t0 = time.perf_counter()
    rng = np.random.RandomState(args.seed)
    data = rng.rand(args.size, args.dim).astype(np.float32)
    np.ascontiguousarray(data).tofile(data_path)
    print(f" data({time.perf_counter() - t0:.1f}s)", end="", flush=True)

    # Write metadata
    meta = {"n_points": args.size, "dim": args.dim, "k": args.k, "dtype": "float32"}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Compute ground truth for requested metrics (skip for large datasets)
    if args.size > MAX_GROUND_TRUTH_POINTS:
        print(" skipping ground truth (n > 100k)", end="", flush=True)
    else:
        for metric in args.truth_metrics:
            truth_path = os.path.join(dirpath, f"truth_{metric}.bin")
            print(f" {metric}", end="", flush=True)
            t0 = time.perf_counter()
            truth = exact_knn(data, args.k, metric)
            np.ascontiguousarray(truth).tofile(truth_path)
            print(f"({time.perf_counter() - t0:.1f}s)", end="", flush=True)

    print(" done")


if __name__ == "__main__":
    main()
