"""Download MNIST and generate subsampled benchmark datasets.

Outputs per size:
    {output_dir}/{n_points}_784/
        meta.json   - {"n_points": N, "dim": 784}
        data.bin    - float32 LE, N x 784, pixel values in [0, 1]
        labels.bin  - uint8, N digit labels (0-9)
"""

import json
import os

import click
import numpy as np

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
SEED = 42


def fetch_mnist():
    """Fetch MNIST (70k images of 28x28 pixels)."""
    from sklearn.datasets import fetch_openml

    print("Fetching MNIST...", flush=True)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    data = mnist["data"].astype(np.float32) / 255.0
    labels = mnist["target"].astype(np.uint8)
    print(f"  Loaded {data.shape[0]} images, shape={data.shape}", flush=True)
    return data, labels


def save_dataset(data, labels, output_dir):
    """Save a dataset to binary files."""
    n, dim = data.shape
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump({"n_points": n, "dim": dim}, f)

    data.astype("<f4").tofile(os.path.join(output_dir, "data.bin"))
    labels.astype(np.uint8).tofile(os.path.join(output_dir, "labels.bin"))
    print(f"  Saved {n} x {dim} to {output_dir}", flush=True)


def generate(output_dir, sizes=None, seed=SEED):
    """Generate MNIST benchmark datasets."""
    sizes = sizes or SIZES
    data, labels = fetch_mnist()
    rng = np.random.RandomState(seed)

    for n in sizes:
        if n > data.shape[0]:
            print(f"Skipping size {n}: only {data.shape[0]} samples available")
            continue

        print(f"\nSubsampling {n} images...", flush=True)
        indices = rng.choice(data.shape[0], size=n, replace=False)
        indices.sort()

        out_dir = os.path.join(output_dir, f"{n}_784")
        save_dataset(data[indices], labels[indices], out_dir)

    print("\nDone.")


@click.command()
@click.option("--sizes", type=int, multiple=True, default=SIZES)
@click.option("--output-dir", required=True)
@click.option("--seed", type=int, default=SEED)
def main(sizes, output_dir, seed):
    generate(output_dir, list(sizes), seed)


if __name__ == "__main__":
    main()
