#!/usr/bin/env python3
"""Orchestrator: generate data, run all benchmarks, collect results.

Usage:
    python benchmark/nndescent/run_all.py [--skip-generate] [--rust-only] [--python-only]

Benchmark matrix:
  sizes:   1k, 2k, 5k, 10k, 20k, 50k, 100k, 200k, 500k
  dims:    100, 200, 400, 800
  metrics: cosine, euclidean

Environment variables:
  BENCH_SINGLE_CPU=1  Run benchmarks on a single CPU (no parallelism).
                      Sets RAYON_NUM_THREADS=1 (Rust) and
                      NUMBA_NUM_THREADS=1 (Python/PyNNDescent).

Results are written to benchmark/results/nndescent/results.csv.
"""

import argparse
import os
import subprocess
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(BENCH_DIR)
PROJECT_DIR = os.path.dirname(BENCHMARK_DIR)

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
DIMS = [100, 200, 400, 800]
METRICS = ["cosine", "euclidean"]
K = 15
SEED = 42

RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results", "nndescent")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
DATA_DIR = os.path.join(BENCHMARK_DIR, "datasets", "nndescent")

SINGLE_CPU = os.environ.get("BENCH_SINGLE_CPU", "0") == "1"


def _bench_env():
    """Return environment dict for benchmark subprocesses."""
    env = os.environ.copy()
    if SINGLE_CPU:
        env["RAYON_NUM_THREADS"] = "1"
        env["NUMBA_NUM_THREADS"] = "1"
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


def generate_data_for(n, dim):
    """Generate benchmark dataset for a single (size, dim) combination."""
    generate_script = os.path.join(BENCH_DIR, "generate_data.py")
    cmd = [
        "uv",
        "run",
        "--with",
        "scikit-learn",
        generate_script,
        "--size",
        str(n),
        "--dim",
        str(dim),
        "--k",
        str(K),
        "--seed",
        str(SEED),
        "--output-dir",
        DATA_DIR,
        "--truth-metrics",
    ] + METRICS

    ok, _ = run_cmd(cmd, f"Generate {n:,} x {dim}")
    if not ok:
        print(f"WARNING: Data generation failed for {n}x{dim}, skipping")
    return ok


def build_rust():
    """Build the Rust benchmark binary in release mode."""
    print("\n" + "=" * 60)
    print("  Build Rust benchmark (release)")
    print("=" * 60)
    cmd = [
        "cargo",
        "build",
        "--release",
        "-p",
        "umap-benchmark",
        "--bin",
        "nndescent-bench",
    ]
    ok, _ = run_cmd(cmd, "Building Rust benchmark", timeout=600)
    if not ok:
        print("ERROR: Rust build failed!")
        sys.exit(1)


def rust_binary_path():
    """Return path to the compiled Rust benchmark binary."""
    return os.path.join(PROJECT_DIR, "target", "release", "nndescent-bench")


def run_benchmarks(run_rust=True, run_python=True, skip_generate=False):
    """Run all benchmark combinations."""
    # Clear results file
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    total = len(SIZES) * len(DIMS) * len(METRICS)
    completed = 0

    for n in SIZES:
        for dim in DIMS:
            data_dir = os.path.join(DATA_DIR, f"{n}_{dim}")

            # Generate data for this (size, dim) just before benchmarking it
            if not skip_generate:
                if not generate_data_for(n, dim):
                    completed += len(METRICS)
                    continue

            if not os.path.exists(data_dir):
                print(f"\nSkipping {n}_{dim}: data not found")
                completed += len(METRICS)
                continue

            for metric in METRICS:
                completed += 1
                label = f"[{completed}/{total}] n={n:,} dim={dim} metric={metric}"

                # Timeout: scale with data size (generous)
                timeout = max(120, n // 500)

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
                        "pynndescent",
                        os.path.join(BENCH_DIR, "benchmark_python.py"),
                        data_dir,
                        metric,
                        "--output",
                        RESULTS_CSV,
                    ]
                    run_cmd(cmd, f"Python {label}", timeout=timeout)


def print_summary():
    """Print a formatted summary table from the results CSV."""
    if not os.path.exists(RESULTS_CSV):
        print("\nNo results file found.")
        return

    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY")
    print("=" * 90)

    with open(RESULTS_CSV) as f:
        lines = f.readlines()

    if len(lines) <= 1:
        print("  No results.")
        return

    rows = [line.strip().split(",") for line in lines[1:] if line.strip()]

    # Group by (n_points, dim, metric)
    from collections import defaultdict

    grouped = defaultdict(dict)
    for row in rows:
        impl_ = row[0]
        key = (row[1], row[2], row[3])  # n_points, dim, metric
        grouped[key][impl_] = row

    print(
        f"\n{'n_points':>10} {'dim':>4} {'metric':>10} | "
        f"{'Rust build':>11} {'Rust recall':>11} | "
        f"{'Py build':>11} {'Py recall':>11} | "
        f"{'Speedup':>8}"
    )
    print("-" * 100)

    for key in sorted(grouped.keys(), key=lambda k: (int(k[0]), int(k[1]), k[2])):
        n, dim, metric = key
        rust = grouped[key].get("rust")
        python = grouped[key].get("python")

        r_build = rust[4] if rust else "—"
        r_recall = rust[5] if rust else "—"
        p_build = python[4] if python else "—"
        p_recall = python[5] if python else "—"

        try:
            speedup = f"{float(p_build) / float(r_build):.1f}x"
        except (ValueError, ZeroDivisionError):
            speedup = "—"

        print(
            f"{n:>10} {dim:>4} {metric:>10} | "
            f"{r_build:>11} {r_recall:>11} | "
            f"{p_build:>11} {p_recall:>11} | "
            f"{speedup:>8}"
        )

    print(f"\nResults saved to: {RESULTS_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument(
        "--skip-generate", action="store_true", help="Skip data generation step"
    )
    parser.add_argument(
        "--rust-only", action="store_true", help="Only run Rust benchmarks"
    )
    parser.add_argument(
        "--python-only", action="store_true", help="Only run Python benchmarks"
    )
    args = parser.parse_args()

    run_rust = not args.python_only
    run_python = not args.rust_only

    if SINGLE_CPU:
        print("\n*** SINGLE-CPU MODE (BENCH_SINGLE_CPU=1) ***\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if run_rust:
        build_rust()

    print("\n" + "=" * 60)
    print("  Running benchmarks")
    print("=" * 60)
    run_benchmarks(
        run_rust=run_rust, run_python=run_python, skip_generate=args.skip_generate
    )

    print_summary()


if __name__ == "__main__":
    main()
