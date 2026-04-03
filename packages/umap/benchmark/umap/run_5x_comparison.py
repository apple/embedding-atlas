#!/usr/bin/env python3
"""Run Rust and Python UMAP benchmarks 5 times each with different seeds,
save all embeddings, and produce a single grid plot (5 rows x 2 columns: Rust | Python).

Usage:
    cd packages/umap
    uv run --with umap-learn,numpy,matplotlib benchmark/umap/run_5x_comparison.py [--data-dir benchmark/datasets/umap/5000_784] [--metric euclidean]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


SEEDS = [42, 123, 456, 789, 1024]


def read_f32_bin(path, n, dim):
    return np.fromfile(path, dtype="<f4").reshape(n, dim)


def run_rust(data_dir, metric, run_idx, rust_bin, seed):
    """Run the Rust UMAP benchmark with a specific seed."""
    emb_path = os.path.join(data_dir, f"rust_{metric}_embedding_run{run_idx}.bin")

    result = subprocess.run(
        [
            rust_bin,
            data_dir,
            metric,
            "--output",
            "/dev/null",
            "--seed",
            str(seed),
            "--embedding-path",
            emb_path,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"  Rust run {run_idx} FAILED:\n{result.stderr}", flush=True)
        return None

    # Parse time from stderr
    for line in result.stderr.splitlines():
        if "UMAP time:" in line:
            elapsed = float(line.split("UMAP time:")[1].strip().rstrip("s"))
            return elapsed
    return None


def run_python(data_dir, metric, run_idx):
    """Run Python UMAP directly (inline) and save embedding."""
    import umap as umap_lib

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    n, dim = meta["n_points"], meta["dim"]

    data = read_f32_bin(os.path.join(data_dir, "data.bin"), n, dim)

    reducer = umap_lib.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric=metric,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    embedding = reducer.fit_transform(data)
    elapsed = time.perf_counter() - t0

    emb_path = os.path.join(data_dir, f"python_{metric}_embedding_run{run_idx}.bin")
    embedding.astype("<f4").tofile(emb_path)
    return elapsed


def _kabsch(emb, ref):
    """Optimal rotation + translation (Kabsch). Returns (transformed, error)."""
    import warnings

    valid = np.isfinite(emb).all(axis=1) & np.isfinite(ref).all(axis=1)
    if valid.sum() < 3:
        return emb.astype(np.float32), np.inf

    e_valid = emb[valid]
    r_valid = ref[valid]

    c_emb = e_valid.mean(axis=0)
    c_ref = r_valid.mean(axis=0)
    E = e_valid - c_emb
    R = r_valid - c_ref

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H = E.T @ R
        U, S, Vt = np.linalg.svd(H)
    if np.any(~np.isfinite(U)) or np.any(~np.isfinite(Vt)):
        return emb.astype(np.float32), np.inf

    d = np.linalg.det(Vt.T @ U.T)
    sign = np.eye(2)
    if d < 0:
        sign[1, 1] = -1
    rot = Vt.T @ sign @ U.T

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (emb - c_emb) @ rot.T + c_ref

    err = np.sum((result[valid] - r_valid) ** 2)
    return result.astype(np.float32), err


def align_to_reference(emb, ref_emb):
    """Align emb to ref_emb using rotation + translation, allowing a flip.

    Tries the embedding as-is and with x-axis flipped, picks whichever
    gives a lower residual after optimal rotation.
    """
    emb = emb.astype(np.float64)
    ref = ref_emb.astype(np.float64)

    result_normal, err_normal = _kabsch(emb, ref)

    flipped = emb.copy()
    flipped[:, 0] = -flipped[:, 0]
    result_flipped, err_flipped = _kabsch(flipped, ref)

    if err_flipped < err_normal:
        return result_flipped
    return result_normal


def make_plot(data_dir, metric, n_runs, n_points, rust_times, python_times):
    """Create a 2-row x n_runs-col grid plot of all embeddings.

    Row 0 = Rust, Row 1 = Python. Python embeddings are rigidly aligned
    to the first Rust embedding so cluster positions are comparable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_points"]

    # Load labels for coloring
    labels_path = os.path.join(data_dir, "labels.bin")
    if os.path.exists(labels_path):
        labels = np.fromfile(labels_path, dtype=np.uint8)
    else:
        labels = np.zeros(n, dtype=np.uint8)

    # Load all embeddings
    rust_embs = []
    python_embs = []
    for run_idx in range(n_runs):
        for impl_name, emb_list in [("rust", rust_embs), ("python", python_embs)]:
            emb_path = os.path.join(
                data_dir, f"{impl_name}_{metric}_embedding_run{run_idx}.bin"
            )
            if os.path.exists(emb_path):
                emb_list.append(read_f32_bin(emb_path, n, 2))
            else:
                emb_list.append(None)

    # Use the first Rust embedding as the reference for alignment
    ref_emb = rust_embs[0]
    if ref_emb is not None:
        # Align all Rust embeddings to the first one
        for i in range(1, len(rust_embs)):
            if rust_embs[i] is not None:
                rust_embs[i] = align_to_reference(rust_embs[i], ref_emb)
        # Align all Python embeddings to the first Rust embedding
        for i in range(len(python_embs)):
            if python_embs[i] is not None:
                python_embs[i] = align_to_reference(python_embs[i], ref_emb)

    fig, axes = plt.subplots(2, n_runs, figsize=(4 * n_runs, 8))
    if n_runs == 1:
        axes = axes.reshape(-1, 1)

    point_size = max(0.5, min(8, 5000 / n))

    for run_idx in range(n_runs):
        for row_idx, (impl_name, emb_list, times) in enumerate(
            [
                ("Rust", rust_embs, rust_times),
                ("Python", python_embs, python_times),
            ]
        ):
            emb = emb_list[run_idx]
            ax = axes[row_idx, run_idx]

            if emb is not None:
                ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=labels,
                    cmap="tab10",
                    s=point_size,
                    alpha=0.6,
                    edgecolors="none",
                    vmin=0,
                    vmax=9,
                )
                ax.set_aspect("equal", adjustable="datalim")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Missing",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )

            t = (
                times[run_idx]
                if run_idx < len(times) and times[run_idx] is not None
                else None
            )
            time_label = f" ({t:.2f}s)" if t is not None else ""

            if run_idx == 0:
                ax.set_ylabel(f"{impl_name}", fontsize=13, fontweight="bold")

            seed_label = SEEDS[run_idx] if run_idx < len(SEEDS) else "?"
            if row_idx == 0:
                ax.set_title(
                    f"Run {run_idx + 1} (seed={seed_label}){time_label}",
                    fontsize=10,
                    fontweight="bold",
                )
            else:
                ax.set_title(f"Run {run_idx + 1}{time_label}", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"UMAP Projections \u2014 {n_points} points, {metric} metric, {n_runs} runs",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path = os.path.join(data_dir, f"5x_comparison_{metric}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out_path}", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run 5x UMAP comparison")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "datasets", "umap", "5000_784"
        ),
        help="Dataset directory (default: benchmark/datasets/umap/5000_784)",
    )
    parser.add_argument("--metric", default="euclidean", help="Distance metric")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of runs")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plot from existing embeddings",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    metric = args.metric
    n_runs = args.n_runs

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    n_points = meta["n_points"]

    # Find Rust binary
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    rust_bin = os.path.join(repo_root, "target", "release", "umap-bench")
    if not os.path.exists(rust_bin):
        print(f"ERROR: Rust binary not found at {rust_bin}", file=sys.stderr)
        print(
            "Build with: cargo build --release -p umap-benchmark --bin umap-bench",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Dataset: {data_dir} ({n_points} points)")
    print(f"Metric: {metric}")
    print(f"Runs: {n_runs}")
    print(f"Seeds: {SEEDS[:n_runs]}")

    if args.plot_only:
        print("\n=== Generating Plot (plot-only mode) ===", flush=True)
        rust_times = [None] * n_runs
        python_times = [None] * n_runs
        make_plot(data_dir, metric, n_runs, n_points, rust_times, python_times)
        return

    print(f"Rust binary: {rust_bin}")
    print()

    # Run Rust benchmarks
    print("=== Rust Benchmarks ===", flush=True)
    rust_times = []
    for i in range(n_runs):
        seed = SEEDS[i]
        print(f"  Run {i + 1}/{n_runs} (seed={seed})...", end=" ", flush=True)
        elapsed = run_rust(data_dir, metric, i, rust_bin, seed)
        rust_times.append(elapsed)
        print(f"{elapsed:.3f}s" if elapsed else "FAILED", flush=True)

    # Run Python benchmarks
    print("\n=== Python Benchmarks ===", flush=True)
    python_times = []
    for i in range(n_runs):
        print(f"  Run {i + 1}/{n_runs}...", end=" ", flush=True)
        elapsed = run_python(data_dir, metric, i)
        python_times.append(elapsed)
        print(f"{elapsed:.3f}s" if elapsed else "FAILED", flush=True)

    # Summary
    valid_rust = [t for t in rust_times if t is not None]
    valid_python = [t for t in python_times if t is not None]
    print("\n=== Summary ===")
    if valid_rust:
        print(
            f"Rust:   mean={np.mean(valid_rust):.3f}s  std={np.std(valid_rust):.3f}s  "
            f"min={np.min(valid_rust):.3f}s  max={np.max(valid_rust):.3f}s"
        )
    if valid_python:
        print(
            f"Python: mean={np.mean(valid_python):.3f}s  std={np.std(valid_python):.3f}s  "
            f"min={np.min(valid_python):.3f}s  max={np.max(valid_python):.3f}s"
        )
    if valid_rust and valid_python:
        print(f"Speedup: {np.mean(valid_python) / np.mean(valid_rust):.1f}x")

    # Generate plot
    print("\n=== Generating Plot ===", flush=True)
    make_plot(data_dir, metric, n_runs, n_points, rust_times, python_times)


if __name__ == "__main__":
    main()
