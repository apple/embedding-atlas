"""Shared utilities for benchmark orchestrators."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import time

# Path constants
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # benchmark/benchmark/
BENCHMARK_DIR = os.path.dirname(PACKAGE_DIR)  # benchmark/
PROJECT_DIR = os.path.dirname(BENCHMARK_DIR)  # packages/umap/
RESULTS_BASE = os.path.join(BENCHMARK_DIR, "results")


def results_dir(suite):
    """Return the results directory for a benchmark suite (umap or nndescent)."""
    return os.path.join(RESULTS_BASE, suite)


def datasets_dir(suite):
    """Return the datasets directory for a benchmark suite."""
    return os.path.join(RESULTS_BASE, suite, "datasets")


def results_csv(suite):
    """Return the results CSV path for a benchmark suite."""
    return os.path.join(RESULTS_BASE, suite, "results.csv")


def report_html(suite):
    """Return the report HTML path for a benchmark suite."""
    return os.path.join(RESULTS_BASE, suite, "report.html")


def rust_binary_path(binary_name):
    """Return path to a compiled Rust benchmark binary."""
    return os.path.join(PROJECT_DIR, "target", "release", binary_name)


def bench_env(threads="multi"):
    """Return environment dict for benchmark subprocesses."""
    env = os.environ.copy()
    if threads == "single":
        env["RAYON_NUM_THREADS"] = "1"
        env["NUMBA_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
    return env


def run_cmd(cmd, desc="", timeout=None, threads="multi"):
    """Run a command, print it, return (success, duration)."""
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    if threads == "single":
        print("  (single-thread mode)")
    print(f"{'=' * 60}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, timeout=timeout, env=bench_env(threads))
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


def run_bench_and_collect(cmd, desc, threads, timeout):
    """Run a benchmark command that writes a JSON result file.

    Returns the parsed result dict with 'threads' added, or None on failure.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name
    try:
        full_cmd = cmd + ["--output", json_path]
        ok, _ = run_cmd(full_cmd, desc, timeout=timeout, threads=threads)
        if not ok:
            return None
        with open(json_path) as f:
            result = json.load(f)
        result["threads"] = threads
        return result
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)


class ResultsCSV:
    """Progressive CSV writer that appends each result as it arrives."""

    def __init__(self, path, columns):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.columns = columns
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=columns, extrasaction="ignore"
        )
        self._writer.writeheader()
        self._file.flush()

    def append(self, row):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


def build_rust(binary_name, gpu=False):
    """Build a Rust benchmark binary in release mode."""
    features = " (with GPU)" if gpu else ""
    cmd = [
        "cargo",
        "build",
        "--release",
        "-p",
        "umap-benchmark",
        "--bin",
        binary_name,
    ]
    if gpu:
        cmd += ["--features", "gpu"]
    ok, _ = run_cmd(cmd, f"Build {binary_name}{features}", timeout=600)
    if not ok:
        print("ERROR: Rust build failed!")
        sys.exit(1)


def python_bench_cmd(module_path, *args):
    """Build a command to run a benchmark Python module."""
    return [sys.executable, "-m", module_path, *args]
