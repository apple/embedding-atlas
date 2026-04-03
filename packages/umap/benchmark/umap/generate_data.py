#!/usr/bin/env python3
"""Download MNIST and generate subsampled benchmark datasets.

Usage:
    uv run --with scikit-learn,numpy bench/generate_data.py [--sizes 1000 2000] [--output-dir bench/results]

Outputs per size:
    {output_dir}/{n_points}_784/
        meta.json   - {"n_points": N, "dim": 784}
        data.bin    - float32 LE, N x 784, pixel values in [0, 1]
        labels.bin  - uint8, N digit labels (0-9)
"""

import argparse
import json
import os

import numpy as np

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
SEED = 42


def fetch_mnist():
    """Fetch MNIST (70k images of 28x28 pixels)."""
    from sklearn.datasets import fetch_openml

    print("Fetching MNIST...", flush=True)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    data = mnist["data"].astype(np.float32) / 255.0  # normalize to [0, 1]
    labels = mnist["target"].astype(np.uint8)
    print(f"  Loaded {data.shape[0]} images, shape={data.shape}", flush=True)
    return data, labels


def save_dataset(data, labels, output_dir):
    """Save a dataset to binary files."""
    n, dim = data.shape
    os.makedirs(output_dir, exist_ok=True)

    # meta.json
    meta = {"n_points": n, "dim": dim}
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # data.bin - float32 little-endian
    data_path = os.path.join(output_dir, "data.bin")
    data.astype("<f4").tofile(data_path)

    # labels.bin - uint8
    labels_path = os.path.join(output_dir, "labels.bin")
    labels.astype(np.uint8).tofile(labels_path)

    print(f"  Saved {n} x {dim} to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate MNIST benchmark data")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=SIZES,
        help="Sample sizes to generate",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    data, labels = fetch_mnist()
    rng = np.random.RandomState(args.seed)

    for n in args.sizes:
        if n > data.shape[0]:
            print(f"Skipping size {n}: only {data.shape[0]} samples available")
            continue

        print(f"\nSubsampling {n} images...", flush=True)
        indices = rng.choice(data.shape[0], size=n, replace=False)
        indices.sort()  # deterministic order

        sub_data = data[indices]
        sub_labels = labels[indices]

        out_dir = os.path.join(args.output_dir, f"{n}_784")
        save_dataset(sub_data, sub_labels, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
