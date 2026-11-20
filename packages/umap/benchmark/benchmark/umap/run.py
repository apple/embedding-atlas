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
from benchmark.umap.python_bench import read_f32_bin
from benchmark.umap.quality import QualityEvaluator

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
METRICS = ["euclidean", "cosine"]

CSV_COLUMNS = [
    "implementation",
    "n_points",
    "dim",
    "metric",
    "gpu",
    "layout",
    "threads",
    "time_s",
    "trustworthiness",
    "knn_recall",
]

LAYOUTS = ["0.5", "0.6"]


def generate_data(sizes):
    """Generate MNIST benchmark datasets."""
    from benchmark.umap.generate_data import generate

    generate(datasets_dir("umap"), sizes)


def collect_with_quality(csv_out, quality, n, cmd, emb_path, desc, threads, timeout):
    """Run one benchmark command and append its timing plus embedding quality."""
    result = run_bench_and_collect(cmd, desc, threads, timeout)
    if result:
        result.update(quality.evaluate(read_f32_bin(emb_path, n, 2)))
        csv_out.append(result)


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

            quality = QualityEvaluator(
                read_f32_bin(os.path.join(data_dir, "data.bin"), n, 784), metric
            )

            for threads in ["single", "multi"]:
                tl = f"({threads}-thread)"

                for layout in LAYOUTS:
                    suffix = "" if layout == "0.5" else "_v06"
                    layout_args = (
                        ["--no-compatibility-layout"] if layout == "0.6" else []
                    )
                    rust_layout_args = (
                        ["--compatibility-layout", "false"] if layout == "0.6" else []
                    )

                    if run_rust:
                        emb_path = os.path.join(
                            data_dir, f"rust{suffix}_{metric}_{threads}_embedding.bin"
                        )
                        cmd = [
                            rust_binary_path("umap-bench"),
                            data_dir,
                            metric,
                            *rust_layout_args,
                            "--embedding-path",
                            emb_path,
                        ]
                        collect_with_quality(
                            csv_out,
                            quality,
                            n,
                            cmd,
                            emb_path,
                            f"Rust {layout} {tl} {label}",
                            threads,
                            timeout,
                        )

                    if run_rust and gpu:
                        emb_path = os.path.join(
                            data_dir,
                            f"rust_gpu{suffix}_{metric}_{threads}_embedding.bin",
                        )
                        cmd = [
                            rust_binary_path("umap-bench"),
                            data_dir,
                            metric,
                            "--gpu",
                            *rust_layout_args,
                            "--embedding-path",
                            emb_path,
                        ]
                        collect_with_quality(
                            csv_out,
                            quality,
                            n,
                            cmd,
                            emb_path,
                            f"Rust GPU {layout} {tl} {label}",
                            threads,
                            timeout,
                        )

                    if run_python:
                        emb_path = os.path.join(
                            data_dir, f"python{suffix}_{metric}_{threads}_embedding.bin"
                        )
                        cmd = python_bench_cmd(
                            "benchmark.umap.python_bench",
                            data_dir,
                            metric,
                            *layout_args,
                            "--embedding-path",
                            emb_path,
                        )
                        collect_with_quality(
                            csv_out,
                            quality,
                            n,
                            cmd,
                            emb_path,
                            f"Python {layout} {tl} {label}",
                            threads,
                            timeout,
                        )

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
        if impl_ == "rust" and row.get("gpu", "false") in ("true", "True"):
            impl_ = "rust-gpu"
        key = (
            row["n_points"],
            row["metric"],
            row.get("threads", "multi"),
            row.get("layout") or "0.5",
        )
        grouped[key][impl_] = row

    hdr = (
        f"{'n_points':>10} {'metric':>10} {'threads':>7} {'umap':>5} | "
        f"{'implementation':>14} {'time_s':>9} {'trust':>7} {'recall':>7}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for key in sorted(grouped, key=lambda k: (int(k[0]), k[1], k[2], k[3])):
        n, metric, threads, layout = key
        for impl_, row in sorted(grouped[key].items()):
            print(
                f"{n:>10} {metric:>10} {threads:>7} {layout:>5} | "
                f"{impl_:>14} {row['time_s']:>9} "
                f"{row.get('trustworthiness') or '—':>7} {row.get('knn_recall') or '—':>7}"
            )

    print(f"\nResults saved to: {csv_path}")
