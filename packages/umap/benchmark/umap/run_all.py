#!/usr/bin/env python3
"""Orchestrator: generate data, run UMAP benchmarks (Rust vs Python), plot results.

Usage:
    python benchmark/umap/run_all.py [--skip-generate] [--rust-only] [--python-only] [--sizes 1000 2000]

Benchmark matrix:
  sizes:   1k, 2k, 5k, 10k, 20k, 50k
  metrics: euclidean, cosine
  UMAP params: n_neighbors=15, n_components=2, min_dist=0.1

Environment variables:
  BENCH_SINGLE_CPU=1  Run benchmarks on a single CPU (no parallelism).
                      Sets RAYON_NUM_THREADS=1 (Rust) and
                      NUMBA_NUM_THREADS=1 / OMP_NUM_THREADS=1 (Python).
"""

import argparse
import os
import subprocess
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(BENCH_DIR)
PROJECT_DIR = os.path.dirname(BENCHMARK_DIR)

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
METRICS = ["euclidean", "cosine"]

DATASETS_DIR = os.path.join(BENCHMARK_DIR, "datasets", "umap")
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results", "umap")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

SINGLE_CPU = os.environ.get("BENCH_SINGLE_CPU", "0") == "1"


def _bench_env():
    """Return environment dict for benchmark subprocesses."""
    env = os.environ.copy()
    if SINGLE_CPU:
        env["RAYON_NUM_THREADS"] = "1"
        env["NUMBA_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
    return env


def run_cmd(cmd, desc="", timeout=None):
    """Run a command, print it, return (success, duration)."""
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    if SINGLE_CPU:
        print("  (single-CPU mode)")
    print(f"{'=' * 60}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, timeout=timeout, env=_bench_env())
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode}) in {elapsed:.1f}s")
            return False, elapsed
        print(f"  OK in {elapsed:.1f}s")
        return True, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return False, elapsed


def generate_data(sizes):
    """Generate benchmark datasets."""
    cmd = (
        [
            "uv",
            "run",
            "--with",
            "scikit-learn,numpy",
            os.path.join(BENCH_DIR, "generate_data.py"),
            "--sizes",
        ]
        + [str(s) for s in sizes]
        + [
            "--output-dir",
            DATASETS_DIR,
        ]
    )
    ok, _ = run_cmd(cmd, "Generate MNIST data", timeout=300)
    if not ok:
        print("ERROR: Data generation failed!")
        sys.exit(1)


def build_rust():
    """Build the Rust UMAP benchmark binary in release mode."""
    cmd = ["cargo", "build", "--release", "-p", "umap-benchmark", "--bin", "umap-bench"]
    ok, _ = run_cmd(cmd, "Build Rust UMAP benchmark", timeout=600)
    if not ok:
        print("ERROR: Rust build failed!")
        sys.exit(1)


def rust_binary_path():
    """Return path to the compiled Rust benchmark binary."""
    return os.path.join(PROJECT_DIR, "target", "release", "umap-bench")


def plot_dataset(data_dir):
    """Generate comparison plots for a single dataset directory."""
    cmd = [
        "uv",
        "run",
        "--with",
        "matplotlib,numpy",
        os.path.join(BENCH_DIR, "plot_results.py"),
        "--results-dir",
        RESULTS_DIR,
        "--data-dir",
        data_dir,
    ]
    run_cmd(cmd, f"Plot {os.path.basename(data_dir)}", timeout=120)


def run_benchmarks(sizes, run_rust=True, run_python=True, skip_plot=False):
    """Run all benchmark combinations."""
    # Clear results CSV
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    total = len(sizes) * len(METRICS)
    completed = 0

    for n in sizes:
        data_dir = os.path.join(DATASETS_DIR, f"{n}_784")

        if not os.path.exists(data_dir):
            print(f"\nSkipping {n}: data not found at {data_dir}")
            completed += len(METRICS)
            continue

        for metric in METRICS:
            completed += 1
            label = f"[{completed}/{total}] n={n:,} metric={metric}"

            # Generous timeout
            timeout = max(300, n // 100)

            # Rust benchmark
            if run_rust:
                cmd = [
                    rust_binary_path(),
                    data_dir,
                    metric,
                    "--output",
                    RESULTS_CSV,
                ]
                run_cmd(cmd, f"Rust {label}", timeout=timeout)

            # Python benchmark
            if run_python:
                cmd = [
                    "uv",
                    "run",
                    "--with",
                    "umap-learn,numpy",
                    os.path.join(BENCH_DIR, "benchmark_python.py"),
                    data_dir,
                    metric,
                    "--output",
                    RESULTS_CSV,
                ]
                run_cmd(cmd, f"Python {label}", timeout=timeout)

        # Plot this dataset immediately after all metrics are done
        if not skip_plot:
            plot_dataset(data_dir)


def plot_timing_chart():
    """Generate the overall timing comparison chart."""
    cmd = [
        "uv",
        "run",
        "--with",
        "matplotlib,numpy",
        os.path.join(BENCH_DIR, "plot_results.py"),
        "--results-dir",
        RESULTS_DIR,
        "--timing-only",
    ]
    run_cmd(cmd, "Generate timing chart", timeout=120)


def print_summary():
    """Print a formatted summary table from the results CSV."""
    if not os.path.exists(RESULTS_CSV):
        print("\nNo results file found.")
        return

    print(f"\n{'=' * 80}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 80}")

    with open(RESULTS_CSV) as f:
        lines = f.readlines()

    if len(lines) <= 1:
        print("  No results.")
        return

    rows = [line.strip().split(",") for line in lines[1:] if line.strip()]

    from collections import defaultdict

    grouped = defaultdict(dict)
    for row in rows:
        impl_ = row[0]
        key = (row[1], row[3])  # n_points, metric
        grouped[key][impl_] = row

    print(
        f"\n{'n_points':>10} {'metric':>10} | "
        f"{'Rust time':>11} | "
        f"{'Python time':>11} | "
        f"{'Speedup':>8}"
    )
    print("-" * 65)

    for key in sorted(grouped.keys(), key=lambda k: (int(k[0]), k[1])):
        n, metric = key
        rust = grouped[key].get("rust")
        python = grouped[key].get("python")

        r_time = rust[4] if rust else "—"
        p_time = python[4] if python else "—"

        try:
            speedup = f"{float(p_time) / float(r_time):.1f}x"
        except (ValueError, ZeroDivisionError):
            speedup = "—"

        print(f"{n:>10} {metric:>10} | {r_time:>11} | {p_time:>11} | {speedup:>8}")

    print(f"\nResults saved to: {RESULTS_CSV}")
    print(f"Plots saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Run all UMAP benchmarks")
    parser.add_argument(
        "--skip-generate", action="store_true", help="Skip data generation"
    )
    parser.add_argument(
        "--rust-only", action="store_true", help="Only run Rust benchmarks"
    )
    parser.add_argument(
        "--python-only", action="store_true", help="Only run Python benchmarks"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=SIZES, help="Sample sizes"
    )
    parser.add_argument("--skip-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    run_rust = not args.python_only
    run_python = not args.rust_only

    if SINGLE_CPU:
        print("\n*** SINGLE-CPU MODE (BENCH_SINGLE_CPU=1) ***\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)

    if not args.skip_generate:
        generate_data(args.sizes)

    if run_rust:
        build_rust()

    run_benchmarks(
        args.sizes, run_rust=run_rust, run_python=run_python, skip_plot=args.skip_plot
    )

    if not args.skip_plot:
        plot_timing_chart()

    print_summary()


if __name__ == "__main__":
    main()
