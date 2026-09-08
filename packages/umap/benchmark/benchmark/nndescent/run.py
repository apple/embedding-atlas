"""NNDescent benchmark orchestrator.

Runs Rust (CPU + GPU) and Python NNDescent benchmarks across sizes, dims,
metrics, and thread modes. Collects results into CSV and generates HTML report.
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
    run_cmd,
    rust_binary_path,
)
from benchmark.html_report import generate_report

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
DIMS = [100, 200, 400, 800]
METRICS = ["cosine", "euclidean"]
K = 15
SEED = 42

CSV_COLUMNS = [
    "implementation",
    "n_points",
    "dim",
    "metric",
    "gpu",
    "threads",
    "build_time_s",
    "recall",
]


def generate_data(sizes, dims):
    """Generate NNDescent benchmark datasets."""
    ds_dir = datasets_dir("nndescent")
    for n in sizes:
        for dim in dims:
            metric_args = []
            for m in METRICS:
                metric_args += ["--truth-metrics", m]
            cmd = python_bench_cmd(
                "benchmark.nndescent.generate_data",
                "--size",
                str(n),
                "--dim",
                str(dim),
                "--k",
                str(K),
                "--seed",
                str(SEED),
                "--output-dir",
                ds_dir,
                *metric_args,
            )
            ok, _ = run_cmd(cmd, f"Generate {n:,} x {dim}")
            if not ok:
                print(f"WARNING: Data generation failed for {n}x{dim}")


def run(
    skip_generate=False,
    skip_python=False,
    python_only=False,
    gpu=True,
    sizes=None,
    dims=None,
):
    """Run all NNDescent benchmarks."""
    sizes = sizes or SIZES
    dims = dims or DIMS
    run_rust = not python_only
    run_python = not skip_python

    ds_dir = datasets_dir("nndescent")
    csv_path = results_csv("nndescent")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(ds_dir, exist_ok=True)

    if not skip_generate:
        generate_data(sizes, dims)

    if run_rust:
        build_rust("nndescent-bench", gpu=gpu)

    total = len(sizes) * len(dims) * len(METRICS)
    completed = 0

    csv_out = ResultsCSV(csv_path, CSV_COLUMNS)

    for n in sizes:
        for dim in dims:
            data_dir = os.path.join(ds_dir, f"{n}_{dim}")

            if not os.path.exists(data_dir):
                print(f"\nSkipping {n}_{dim}: data not found")
                completed += len(METRICS)
                continue

            for metric in METRICS:
                completed += 1
                label = f"[{completed}/{total}] n={n:,} dim={dim} metric={metric}"
                timeout = max(120, n // 500)

                for threads in ["single", "multi"]:
                    tl = f"({threads}-thread)"

                    if run_rust:
                        cmd = [rust_binary_path("nndescent-bench"), data_dir, metric]
                        r = run_bench_and_collect(
                            cmd, f"Rust {tl} {label}", threads, timeout
                        )
                        if r:
                            csv_out.append(r)

                    if run_rust and gpu:
                        cmd = [
                            rust_binary_path("nndescent-bench"),
                            data_dir,
                            metric,
                            "--gpu",
                        ]
                        r = run_bench_and_collect(
                            cmd, f"Rust GPU {tl} {label}", threads, timeout
                        )
                        if r:
                            csv_out.append(r)

                    if run_python:
                        cmd = python_bench_cmd(
                            "benchmark.nndescent.python_bench", data_dir, metric
                        )
                        r = run_bench_and_collect(
                            cmd, f"Python {tl} {label}", threads, timeout
                        )
                        if r:
                            csv_out.append(r)

    csv_out.close()
    generate_report(csv_path, report_html("nndescent"), "nndescent")
    print_summary(csv_path)


def print_summary(csv_path):
    """Print a formatted summary table."""
    import csv as csv_mod
    from collections import defaultdict

    if not os.path.exists(csv_path):
        print("\nNo results file found.")
        return

    print(f"\n{'=' * 110}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 110}")

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
        key = (row["n_points"], row["dim"], row["metric"], row.get("threads", "multi"))
        grouped[key][impl_] = (row["build_time_s"], row["recall"])

    has_gpu = any("rust-gpu" in v for v in grouped.values())

    hdr = f"{'n_points':>10} {'dim':>4} {'metric':>10} {'threads':>7} | {'Rust build':>11} {'Rust recall':>11}"
    if has_gpu:
        hdr += f" | {'GPU build':>11} {'GPU recall':>11}"
    hdr += f" | {'Py build':>11} {'Py recall':>11} | {'Speedup':>8}"
    if has_gpu:
        hdr += f" {'GPU spd':>8}"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for key in sorted(grouped.keys(), key=lambda k: (int(k[0]), int(k[1]), k[2], k[3])):
        n, dim, metric, threads = key
        rust = grouped[key].get("rust")
        rust_gpu = grouped[key].get("rust-gpu")
        python = grouped[key].get("python")

        r_build = rust[0] if rust else "\u2014"
        r_recall = rust[1] if rust else "\u2014"
        g_build = rust_gpu[0] if rust_gpu else "\u2014"
        g_recall = rust_gpu[1] if rust_gpu else "\u2014"
        p_build = python[0] if python else "\u2014"
        p_recall = python[1] if python else "\u2014"

        try:
            speedup = f"{float(p_build) / float(r_build):.1f}x"
        except (ValueError, ZeroDivisionError):
            speedup = "\u2014"
        try:
            gpu_speedup = f"{float(p_build) / float(g_build):.1f}x"
        except (ValueError, ZeroDivisionError):
            gpu_speedup = "\u2014"

        line = (
            f"{n:>10} {dim:>4} {metric:>10} {threads:>7} | {r_build:>11} {r_recall:>11}"
        )
        if has_gpu:
            line += f" | {g_build:>11} {g_recall:>11}"
        line += f" | {p_build:>11} {p_recall:>11} | {speedup:>8}"
        if has_gpu:
            line += f" {gpu_speedup:>8}"
        print(line)

    print(f"\nResults saved to: {csv_path}")
