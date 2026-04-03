#!/usr/bin/env python3
"""Generate comparison scatter plots for UMAP benchmark results.

Usage:
    uv run --with matplotlib,numpy bench/plot_results.py [--results-dir bench/results]

For each dataset and metric, creates side-by-side scatter plots colored by digit class.
Also generates a timing comparison chart.
"""

import argparse
import csv
import json
import os

import numpy as np


def read_f32_bin(path, n, ncols):
    """Read float32 LE binary file."""
    return np.fromfile(path, dtype="<f4").reshape(n, ncols)


def read_labels(path, n):
    """Read uint8 labels."""
    return np.fromfile(path, dtype=np.uint8)[:n]


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


def plot_comparison(data_dir, metric, results_csv_path):
    """Create comparison scatter plot for one dataset/metric pair.

    Layout: 2 rows (Rust, Python) x 1 column.
    Python embedding is rigidly aligned to the Rust embedding so cluster
    positions are comparable.
    """
    import matplotlib.pyplot as plt

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_points"]

    labels_path = os.path.join(data_dir, "labels.bin")
    if not os.path.exists(labels_path):
        print(f"  Skipping {data_dir}: no labels.bin")
        return

    labels = read_labels(labels_path, n)

    rust_path = os.path.join(data_dir, f"rust_{metric}_embedding.bin")
    python_path = os.path.join(data_dir, f"python_{metric}_embedding.bin")

    has_rust = os.path.exists(rust_path)
    has_python = os.path.exists(python_path)

    if not has_rust and not has_python:
        print(f"  Skipping {data_dir}/{metric}: no embeddings")
        return

    # Read timing from results CSV
    rust_time = None
    python_time = None
    if os.path.exists(results_csv_path):
        with open(results_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["n_points"]) == n and row["metric"] == metric:
                    if row["implementation"] == "rust":
                        rust_time = float(row["time_s"])
                    elif row["implementation"] == "python":
                        python_time = float(row["time_s"])

    # Load embeddings
    rust_emb = read_f32_bin(rust_path, n, 2) if has_rust else None
    python_emb = read_f32_bin(python_path, n, 2) if has_python else None

    # Align Python embedding to Rust embedding
    if rust_emb is not None and python_emb is not None:
        python_emb = align_to_reference(python_emb, rust_emb)

    nrows = (1 if has_rust else 0) + (1 if has_python else 0)
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 6 * nrows))
    if nrows == 1:
        axes = [axes]

    point_size = max(0.5, min(8, 5000 / n))

    row = 0
    for impl_name, emb, t in [
        ("Rust", rust_emb, rust_time),
        ("Python", python_emb, python_time),
    ]:
        if emb is None:
            continue
        ax = axes[row]
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
        time_label = f" ({t:.1f}s)" if t is not None else ""
        ax.set_ylabel(impl_name, fontsize=13, fontweight="bold")
        ax.set_title(f"{impl_name} UMAP (n={n:,}, {metric}){time_label}")
        ax.set_xticks([])
        ax.set_yticks([])
        row += 1

    fig.tight_layout()
    out_path = os.path.join(data_dir, f"comparison_{metric}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_timing(results_csv_path, output_dir):
    """Generate timing comparison bar chart."""
    import matplotlib.pyplot as plt

    if not os.path.exists(results_csv_path):
        return

    with open(results_csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    # Group by (n_points, metric)
    from collections import defaultdict

    grouped = defaultdict(dict)
    for row in rows:
        key = (int(row["n_points"]), row["metric"])
        grouped[key][row["implementation"]] = float(row["time_s"])

    keys = sorted(grouped.keys())
    if not keys:
        return

    # Separate by metric
    metrics = sorted(set(k[1] for k in keys))

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        metric_keys = [k for k in keys if k[1] == metric]
        sizes = [k[0] for k in metric_keys]
        rust_times = [grouped[k].get("rust", 0) for k in metric_keys]
        python_times = [grouped[k].get("python", 0) for k in metric_keys]

        x = np.arange(len(sizes))
        width = 0.35

        ax.bar(x - width / 2, rust_times, width, label="Rust", color="tab:blue")
        ax.bar(x + width / 2, python_times, width, label="Python", color="tab:orange")
        ax.set_xlabel("Number of points")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"UMAP Timing - {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=45, ha="right")
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "timing_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot UMAP benchmark results")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Results directory",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Plot only this dataset directory (skip others)",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Only generate the timing comparison chart",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    results_csv = os.path.join(results_dir, "results.csv")

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    metrics = ["euclidean", "cosine"]

    if not args.timing_only:
        # Plot per-dataset comparisons
        if args.data_dir:
            # Plot a single dataset directory
            data_dir = os.path.abspath(args.data_dir)
            if os.path.exists(os.path.join(data_dir, "meta.json")):
                print(f"\nPlotting {os.path.basename(data_dir)}...")
                for metric in metrics:
                    plot_comparison(data_dir, metric, results_csv)
        else:
            # Plot all datasets
            for entry in sorted(os.listdir(results_dir)):
                entry_path = os.path.join(results_dir, entry)
                if not os.path.isdir(entry_path):
                    continue
                if not os.path.exists(os.path.join(entry_path, "meta.json")):
                    continue

                print(f"\nPlotting {entry}...")
                for metric in metrics:
                    plot_comparison(entry_path, metric, results_csv)

    # Timing comparison chart
    if not args.data_dir:
        print("\nGenerating timing comparison chart...")
        plot_timing(results_csv, results_dir)


if __name__ == "__main__":
    main()
