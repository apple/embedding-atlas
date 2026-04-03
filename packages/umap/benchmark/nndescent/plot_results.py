#!/usr/bin/env python3
"""Plot benchmark results from results.csv as faceted line charts."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(BENCH_DIR)
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results", "nndescent")

df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))

metrics = sorted(df["metric"].unique())
dims = sorted(df["dim"].unique())

# Rows: metric, Cols: dim. Each cell has two y-axes (time + recall).
fig, axes = plt.subplots(
    len(metrics),
    len(dims),
    figsize=(5 * len(dims), 4 * len(metrics)),
    squeeze=False,
)

colors = {"rust": "#e24a33", "python": "#348abd"}
styles = {"rust": "-o", "python": "--s"}

for row, metric in enumerate(metrics):
    for col, dim in enumerate(dims):
        ax_time = axes[row][col]
        ax_recall = ax_time.twinx()

        sub = df[(df["metric"] == metric) & (df["dim"] == dim)]

        for impl in ["rust", "python"]:
            grp = sub[sub["implementation"] == impl].sort_values("n_points")
            ax_time.plot(
                grp["n_points"],
                grp["build_time_s"],
                styles[impl],
                color=colors[impl],
                label=f"{impl} (time)",
                markersize=5,
                linewidth=1.5,
            )
            ax_recall.plot(
                grp["n_points"],
                grp["recall"],
                styles[impl],
                color=colors[impl],
                label=f"{impl} (recall)",
                markersize=5,
                linewidth=1.5,
                alpha=0.45,
            )

        ax_time.set_xscale("log")
        ax_time.set_title(f"{metric} | dim={dim}", fontsize=11, fontweight="bold")
        ax_time.set_xlabel("n_points")
        ax_time.grid(True, alpha=0.2)

        if col == 0:
            ax_time.set_ylabel("build_time_s", color="#555")
        if col == len(dims) - 1:
            ax_recall.set_ylabel("recall", color="#555")
        else:
            ax_recall.set_yticklabels([])


legend_elements = [
    Line2D(
        [0], [0], color=colors["rust"], marker="o", linestyle="-", label="rust (time)"
    ),
    Line2D(
        [0],
        [0],
        color=colors["rust"],
        marker="o",
        linestyle="-",
        alpha=0.45,
        label="rust (recall)",
    ),
    Line2D(
        [0],
        [0],
        color=colors["python"],
        marker="s",
        linestyle="--",
        label="python (time)",
    ),
    Line2D(
        [0],
        [0],
        color=colors["python"],
        marker="s",
        linestyle="--",
        alpha=0.45,
        label="python (recall)",
    ),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.02),
)

out_path = os.path.join(RESULTS_DIR, "results.png")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
