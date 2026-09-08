"""Generate random benchmark datasets with ground truth for NNDescent.

Outputs per (size, dim):
    {output_dir}/{n_points}_{dim}/
        meta.json        - {"n_points": N, "dim": D, "k": K, "dtype": "float32"}
        data.bin         - float32 LE, N x D
        truth_{metric}.bin - int32 LE, N x K brute-force NN indices
"""

import json
import os
import time

import click
import numpy as np

MAX_GROUND_TRUTH_POINTS = 100_000
K = 15
SEED = 42


def exact_knn(data, k, metric):
    """Compute exact k-NN using sklearn, including self-matches."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    data64 = data.astype(np.float64)

    if metric == "cosine":
        data64 = normalize(data64, norm="l2")
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    else:
        nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")

    nn.fit(data64)
    _, indices = nn.kneighbors(data64)
    return indices.astype(np.int32)


def generate_one(size, dim, output_dir, truth_metrics, k=K, seed=SEED):
    """Generate a single dataset with ground truth."""
    dirname = f"{size}_{dim}"
    dirpath = os.path.join(output_dir, dirname)
    os.makedirs(dirpath, exist_ok=True)

    meta_path = os.path.join(dirpath, "meta.json")
    data_path = os.path.join(dirpath, "data.bin")

    # Check if already generated
    if os.path.exists(meta_path) and os.path.exists(data_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if (
            meta.get("n_points") == size
            and meta.get("dim") == dim
            and meta.get("k") == k
        ):
            all_truth_exist = size > MAX_GROUND_TRUTH_POINTS or all(
                os.path.exists(os.path.join(dirpath, f"truth_{m}.bin"))
                and os.path.getsize(os.path.join(dirpath, f"truth_{m}.bin")) > 0
                for m in truth_metrics
            )
            if all_truth_exist:
                print(f"{dirname}: already exists, skipping")
                return

    print(f"{dirname}: generating {size} x {dim} ...", end="", flush=True)
    t0 = time.perf_counter()
    rng = np.random.RandomState(seed)
    data = rng.rand(size, dim).astype(np.float32)
    np.ascontiguousarray(data).tofile(data_path)
    print(f" data({time.perf_counter() - t0:.1f}s)", end="", flush=True)

    meta = {"n_points": size, "dim": dim, "k": k, "dtype": "float32"}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if size > MAX_GROUND_TRUTH_POINTS:
        print(" skipping ground truth (n > 100k)", end="", flush=True)
    else:
        for metric in truth_metrics:
            truth_path = os.path.join(dirpath, f"truth_{metric}.bin")
            print(f" {metric}", end="", flush=True)
            t0 = time.perf_counter()
            truth = exact_knn(data, k, metric)
            np.ascontiguousarray(truth).tofile(truth_path)
            print(f"({time.perf_counter() - t0:.1f}s)", end="", flush=True)

    print(" done")


@click.command()
@click.option("--size", type=int, required=True)
@click.option("--dim", type=int, required=True)
@click.option("--k", type=int, default=K)
@click.option("--seed", type=int, default=SEED)
@click.option("--truth-metrics", multiple=True, default=[])
@click.option("--output-dir", required=True)
def main(size, dim, k, seed, truth_metrics, output_dir):
    generate_one(size, dim, output_dir, list(truth_metrics), k=k, seed=seed)


if __name__ == "__main__":
    main()
