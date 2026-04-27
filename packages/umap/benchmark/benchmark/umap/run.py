"""UMAP benchmark orchestrator.

Runs Rust (CPU + GPU) and Python UMAP benchmarks across sizes, metrics,
and thread modes. Collects results into CSV and generates HTML report.
"""

import os

from benchmark.common import (
    ResultsCSV,
    build_rust,
    datasets_dir,
    python_bench_cmd,
    report_html,
    results_csv,
    run_bench_and_collect,
    rust_binary_path,
)
from benchmark.html_report import generate_report

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
METRICS = ["euclidean", "cosine"]

CSV_COLUMNS = [
    "implementation",
    "n_points",
    "dim",
    "metric",
    "gpu",
    "threads",
    "time_s",
]


def generate_data(sizes):
    """Generate MNIST benchmark datasets."""
    from benchmark.umap.generate_data import generate

    generate(datasets_dir("umap"), sizes)


def run(
    skip_generate=False,
    skip_python=False,
    python_only=False,
    gpu=True,
    skip_plot=False,
    sizes=None,
):
    """Run all UMAP benchmarks."""
    sizes = sizes or SIZES
    run_rust = not python_only
    run_python = not skip_python

    ds_dir = datasets_dir("umap")
    csv_path = results_csv("umap")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(ds_dir, exist_ok=True)

    if not skip_generate:
        generate_data(sizes)

    if run_rust:
        build_rust("umap-bench", gpu=gpu)

    total = len(sizes) * len(METRICS)
    completed = 0

    csv_out = ResultsCSV(csv_path, CSV_COLUMNS)

    for n in sizes:
        data_dir = os.path.join(ds_dir, f"{n}_784")

        if not os.path.exists(data_dir):
            print(f"\nSkipping {n}: data not found at {data_dir}")
            completed += len(METRICS)
            continue

        for metric in METRICS:
            completed += 1
            label = f"[{completed}/{total}] n={n:,} metric={metric}"
            timeout = max(300, n // 100)

            for threads in ["single", "multi"]:
                tl = f"({threads}-thread)"

                if run_rust:
                    emb_path = os.path.join(
                        data_dir, f"rust_{metric}_{threads}_embedding.bin"
                    )
                    cmd = [
                        rust_binary_path("umap-bench"),
                        data_dir,
                        metric,
                        "--embedding-path",
                        emb_path,
                    ]
                    r = run_bench_and_collect(
                        cmd, f"Rust {tl} {label}", threads, timeout
                    )
                    if r:
                        csv_out.append(r)

                if run_rust and gpu:
                    emb_path = os.path.join(
                        data_dir, f"rust_gpu_{metric}_{threads}_embedding.bin"
                    )
                    cmd = [
                        rust_binary_path("umap-bench"),
                        data_dir,
                        metric,
                        "--gpu",
                        "--embedding-path",
                        emb_path,
                    ]
                    r = run_bench_and_collect(
                        cmd, f"Rust GPU {tl} {label}", threads, timeout
                    )
                    if r:
                        csv_out.append(r)

                if run_python:
                    emb_path = os.path.join(
                        data_dir, f"python_{metric}_{threads}_embedding.bin"
                    )
                    cmd = python_bench_cmd(
                        "benchmark.umap.python_bench",
                        data_dir,
                        metric,
                        "--embedding-path",
                        emb_path,
                    )
                    r = run_bench_and_collect(
                        cmd, f"Python {tl} {label}", threads, timeout
                    )
                    if r:
                        csv_out.append(r)

    csv_out.close()

    if not skip_plot:
        generate_report(
            csv_path, report_html("umap"), "umap", datasets_dir=datasets_dir("umap")
        )

    print_summary(csv_path)


def print_summary(csv_path):
    """Print a formatted summary table."""
    import csv as csv_mod
    from collections import defaultdict

    if not os.path.exists(csv_path):
        print("\nNo results file found.")
        return

    print(f"\n{'=' * 80}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 80}")

    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    if not rows:
        print("  No results.")
        return

    grouped = defaultdict(dict)
    for row in rows:
        impl_ = row["implementation"]
        gpu_val = row.get("gpu", "false")
        if impl_ == "rust" and gpu_val in ("true", "True"):
            impl_ = "rust-gpu"
        key = (row["n_points"], row["metric"], row.get("threads", "multi"))
        grouped[key][impl_] = row["time_s"]

    has_gpu = any("rust-gpu" in v for v in grouped.values())

    hdr = f"{'n_points':>10} {'metric':>10} {'threads':>7} | {'Rust time':>11}"
    if has_gpu:
        hdr += f" {'GPU time':>11}"
    hdr += f" | {'Python time':>11} | {'Speedup':>8}"
    if has_gpu:
        hdr += f" {'GPU spd':>8}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for key in sorted(grouped.keys(), key=lambda k: (int(k[0]), k[1], k[2])):
        n, metric, threads = key
        r_time = grouped[key].get("rust", "—")
        g_time = grouped[key].get("rust-gpu", "—")
        p_time = grouped[key].get("python", "—")

        try:
            speedup = f"{float(p_time) / float(r_time):.1f}x"
        except (ValueError, ZeroDivisionError):
            speedup = "—"
        try:
            gpu_speedup = f"{float(p_time) / float(g_time):.1f}x"
        except (ValueError, ZeroDivisionError):
            gpu_speedup = "—"

        line = f"{n:>10} {metric:>10} {threads:>7} | {r_time:>11}"
        if has_gpu:
            line += f" {g_time:>11}"
        line += f" | {p_time:>11} | {speedup:>8}"
        if has_gpu:
            line += f" {gpu_speedup:>8}"
        print(line)

    print(f"\nResults saved to: {csv_path}")
