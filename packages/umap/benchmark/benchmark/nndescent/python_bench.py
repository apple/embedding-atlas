"""Python NNDescent benchmark using pynndescent.

Reads meta.json + data.bin, builds NNDescent index, computes recall
against ground truth, and writes a JSON result file.
"""

import json
import os
import sys
import time

import click
import numpy as np


def compute_recall(true_indices, approx_indices):
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


@click.command()
@click.argument("data_dir")
@click.argument("metric")
@click.option("--output", default=None, help="JSON output path")
def main(data_dir, metric, output):
    output_path = output or os.path.join(data_dir, "result.json")

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    n, dim, k = meta["n_points"], meta["dim"], meta["k"]

    print(
        f"Python NNDescent benchmark: n={n}, dim={dim}, k={k}, metric={metric}",
        file=sys.stderr,
        flush=True,
    )

    print("  Loading data...", file=sys.stderr, flush=True)
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

    single_cpu = os.environ.get("NUMBA_NUM_THREADS", "") == "1"
    n_jobs_val = 1 if single_cpu else -1
    if single_cpu:
        print("  Single-CPU mode: n_jobs=1", file=sys.stderr, flush=True)

    # Warm up Numba JIT
    print("  Warming up Numba JIT...", file=sys.stderr, flush=True)
    warmup_data = np.random.RandomState(0).rand(100, dim).astype(np.float32)
    _ = NNDescent(
        warmup_data,
        metric=metric,
        n_neighbors=min(k, 10),
        random_state=0,
        n_jobs=n_jobs_val,
    )
    del warmup_data, _

    # Build index
    print(f"  Building index on {n} points...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    nnd = NNDescent(
        data, metric=metric, n_neighbors=k, random_state=42, n_jobs=n_jobs_val
    )
    build_time = time.perf_counter() - t0
    print(f"  Build time: {build_time:.3f}s", file=sys.stderr, flush=True)

    # Compute recall
    recall_str = "N/A"
    if truth is not None:
        approx_indices = nnd.neighbor_graph[0].copy()
        for i in range(n):
            row = approx_indices[i]
            positions = np.where(row == i)[0]
            if len(positions) > 0 and positions[0] == 0:
                continue
            if len(positions) > 0:
                pos = positions[0]
                approx_indices[i, 1 : pos + 1] = approx_indices[i, 0:pos]
            else:
                approx_indices[i, 1:] = approx_indices[i, :-1]
            approx_indices[i, 0] = i
        recall = compute_recall(truth, approx_indices)
        print(f"  Recall: {recall:.4f}", file=sys.stderr, flush=True)
        recall_str = f"{recall:.4f}"
    else:
        print("  No ground truth, skipping recall", file=sys.stderr, flush=True)

    result = {
        "implementation": "python",
        "n_points": n,
        "dim": dim,
        "metric": metric,
        "gpu": False,
        "build_time_s": round(build_time, 3),
        "recall": recall_str,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"  Result written to {output_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
