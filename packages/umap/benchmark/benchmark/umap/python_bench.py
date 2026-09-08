"""Python UMAP benchmark using umap-learn.

Reads meta.json + data.bin, runs UMAP with matching parameters,
saves embedding and writes a JSON result file.
"""

import json
import os
import time

import click
import numpy as np


def read_f32_bin(path, n, dim):
    return np.fromfile(path, dtype="<f4").reshape(n, dim)


def write_f32_bin(path, data):
    data.astype("<f4").tofile(path)


@click.command()
@click.argument("data_dir")
@click.argument("metric")
@click.option("--output", default=None, help="JSON output path")
@click.option("--embedding-path", default=None, help="Embedding output path")
def main(data_dir, metric, output, embedding_path):
    output_path = output or os.path.join(data_dir, "result.json")

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_points"]
    dim = meta["dim"]

    print(f"Python UMAP benchmark: n={n}, dim={dim}, metric={metric}", flush=True)

    print("  Loading data...", flush=True)
    data = read_f32_bin(os.path.join(data_dir, "data.bin"), n, dim)

    import umap as umap_lib

    print("  Running UMAP...", flush=True)
    single_cpu = os.environ.get("OMP_NUM_THREADS", "") == "1"
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

    emb_path = embedding_path or os.path.join(
        data_dir, f"python_{metric}_embedding.bin"
    )
    write_f32_bin(emb_path, embedding.astype(np.float32))
    print(f"  Saved embedding to {emb_path}", flush=True)

    result = {
        "implementation": "python",
        "n_points": n,
        "dim": dim,
        "metric": metric,
        "gpu": False,
        "time_s": round(elapsed, 3),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"  Result written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
