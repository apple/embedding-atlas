"""Markdown report for the UMAP parameter sweep."""

import numpy as np
import pandas as pd

from benchmark.common import RESULTS_BASE
from benchmark.umap.sweep import CSV_PATH
from benchmark.umap.sweep_config import (
    DATASETS,
    MAIN_SEED,
    MIN_SEEDS_FOR_MEAN_RULE,
    VARIANTS,
    python_partner,
)

SUMMARY_PATH = RESULTS_BASE / "sweep_summary.md"
MIN_TRUST_THRESHOLD = 0.01
RECALL_THRESHOLD = 0.02
COLLAPSE_SPREAD = 1e-3

RUST_VARIANTS = [v for v in VARIANTS if v.implementation == "rust"]
GPU_TO_CPU = {
    "rust_05_sgd_gpu": "rust_05_sgd_cpu",
    "rust_06_adam_gpu": "rust_06_adam_cpu",
}
NUMERIC = [
    "time_s",
    "knn_s",
    "layout_s",
    "trustworthiness",
    "knn_recall",
    "spread_min",
    "spread_max",
    "procrustes",
]


def load_results():
    df = pd.read_csv(CSV_PATH, keep_default_na=False, na_values=[""])
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["config"] = df["config"].astype(str)
    return df


def seed_noise(df):
    """Std of trustworthiness over the default-config seeds, per (dataset, variant)."""
    default = df[(df.config == "default") & (df.status == "ok")]
    return (
        default.groupby(["dataset", "variant"])["trustworthiness"]
        .agg(["std", "count"])
        .query("count > 1")["std"]
        .to_dict()
    )


def trust_threshold(noise, dataset, rust_variant, python_variant):
    std = max(
        noise.get((dataset, rust_variant), 0.0),
        noise.get((dataset, python_variant), 0.0),
    )
    return max(MIN_TRUST_THRESHOLD, 3 * float(np.nan_to_num(std)))


def main_rows(df):
    return df[df.seed == MAIN_SEED].set_index(["dataset", "config", "variant"])


def seed_stats(df):
    """Per (dataset, config, variant): mean and std over seeds of the finite ok runs."""
    ok = df[(df.status == "ok") & df.trustworthiness.notna()]
    grouped = (
        ok.groupby(["dataset", "config", "variant"])
        .agg(
            n=("seed", "count"),
            trust_mean=("trustworthiness", "mean"),
            trust_std=("trustworthiness", "std"),
            recall_mean=("knn_recall", "mean"),
            recall_std=("knn_recall", "std"),
        )
        .fillna(0.0)
    )
    return {key: row for key, row in grouped.iterrows()}


def find_flags(df):
    rows = main_rows(df)
    noise = seed_noise(df)
    flags = []

    def get(dataset, config, variant):
        return (
            rows.loc[(dataset, config, variant)]
            if (dataset, config, variant) in rows.index
            else None
        )

    for dataset, config, variant in rows.index:
        row = rows.loc[(dataset, config, variant)]
        if row.status == "ok" and (not row.finite or row.spread_min < COLLAPSE_SPREAD):
            kind = "NaN/inf" if not row.finite else "collapsed"
            kind += f" ({row.implementation})"
            flags.append(
                dict(
                    kind=kind,
                    dataset=dataset,
                    config=config,
                    variant=variant,
                    detail=f"spread_min={row.spread_min}",
                )
            )

    stats = seed_stats(df)

    def compare(dataset, config, a_name, b_name):
        """Return (d_trust, d_recall, trust_limit, rule) for variant a minus variant b, or None."""
        a, b = (
            stats.get((dataset, config, a_name)),
            stats.get((dataset, config, b_name)),
        )
        if a is None or b is None:
            return None
        if min(a.n, b.n) >= MIN_SEEDS_FOR_MEAN_RULE:
            pooled = np.sqrt((a.trust_std**2 + b.trust_std**2) / 2)
            limit = max(MIN_TRUST_THRESHOLD, 2 * pooled / np.sqrt(min(a.n, b.n)))
            rule = f"mean of {min(a.n, b.n)} seeds"
            return (
                a.trust_mean - b.trust_mean,
                a.recall_mean - b.recall_mean,
                limit,
                rule,
            )
        ra, rb = get(dataset, config, a_name), get(dataset, config, b_name)
        limit = trust_threshold(noise, dataset, a_name, b_name)
        return (
            ra.trustworthiness - rb.trustworthiness,
            ra.knn_recall - rb.knn_recall,
            limit,
            "seed 42",
        )

    configs = (
        rows.reset_index()[["dataset", "config"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    for dataset, config in configs:
        for variant in RUST_VARIANTS:
            rust = get(dataset, config, variant.name)
            python = get(dataset, config, python_partner(variant))
            if rust is None or python is None:
                continue
            if rust.status != "ok":
                if python.status == "ok":
                    flags.append(
                        dict(
                            kind="rust error",
                            dataset=dataset,
                            config=config,
                            variant=variant.name,
                            detail=str(rust.reason),
                        )
                    )
                continue
            comparisons = [("worse than Python", python_partner(variant))]
            if variant.name in GPU_TO_CPU:
                comparisons.append(("GPU worse than CPU", GPU_TO_CPU[variant.name]))
            for kind, other in comparisons:
                result = compare(dataset, config, variant.name, other)
                if result is None:
                    continue
                d_trust, d_recall, limit, rule = result
                if d_trust < -limit or d_recall < -RECALL_THRESHOLD:
                    flags.append(
                        dict(
                            kind=kind,
                            dataset=dataset,
                            config=config,
                            variant=variant.name,
                            detail=f"dTrust={d_trust:+.4f} (limit -{limit:.4f}), dRecall={d_recall:+.4f} [{rule}]",
                        )
                    )
    return pd.DataFrame(
        flags, columns=["kind", "dataset", "config", "variant", "detail"]
    )


def delta_table(df, flags):
    """One row per (dataset, config); one cell per Rust variant with the Rust minus Python deltas."""
    rows = main_rows(df)
    flagged = {(f.dataset, f.config, f.variant) for f in flags.itertuples()}
    lines = []
    keys = (
        rows.reset_index()[["dataset", "config"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    for dataset, config in keys:
        cells = []
        for variant in RUST_VARIANTS:
            rust_key = (dataset, config, variant.name)
            python_key = (dataset, config, python_partner(variant))
            if rust_key not in rows.index or python_key not in rows.index:
                cells.append("n/a")
                continue
            rust, python = rows.loc[rust_key], rows.loc[python_key]
            if rust.status != "ok":
                cells.append("ERR")
            elif python.status != "ok":
                cells.append("py rejects")
            elif not (
                np.isfinite(rust.trustworthiness)
                and np.isfinite(python.trustworthiness)
            ):
                cells.append(
                    "NaN ("
                    + ("Rust" if not np.isfinite(rust.trustworthiness) else "Python")
                    + ")"
                )
            else:
                text = (
                    f"{rust.trustworthiness - python.trustworthiness:+.3f}/"
                    f"{rust.knn_recall - python.knn_recall:+.3f}"
                )
                cells.append(f"**{text}**" if rust_key in flagged else text)
        lines.append((dataset, config, cells))
    return lines


def overall_table(df):
    rows = main_rows(df)
    out = []
    for dataset in [d.name for d in DATASETS]:
        for variant in RUST_VARIANTS:
            deltas = []
            for config in (
                rows.loc[dataset].index.get_level_values("config").unique()
                if dataset in rows.index.get_level_values(0)
                else []
            ):
                rust_key, python_key = (
                    (dataset, config, variant.name),
                    (dataset, config, python_partner(variant)),
                )
                if rust_key in rows.index and python_key in rows.index:
                    rust, python = rows.loc[rust_key], rows.loc[python_key]
                    if (
                        rust.status == "ok"
                        and python.status == "ok"
                        and np.isfinite(rust.trustworthiness)
                        and np.isfinite(python.trustworthiness)
                    ):
                        deltas.append(
                            (
                                rust.trustworthiness - python.trustworthiness,
                                rust.knn_recall - python.knn_recall,
                                rust.procrustes,
                            )
                        )
            if deltas:
                arr = np.array(deltas, dtype=float)
                out.append(
                    dict(
                        dataset=dataset,
                        variant=variant.name,
                        n=len(arr),
                        mean_dtrust=arr[:, 0].mean(),
                        min_dtrust=arr[:, 0].min(),
                        mean_drecall=arr[:, 1].mean(),
                        min_drecall=arr[:, 1].min(),
                        median_procrustes=np.nanmedian(arr[:, 2]),
                    )
                )
    return pd.DataFrame(out)


def pipeline_totals(overall):
    totals = overall.copy()
    totals["pipeline"] = totals.variant.map(
        {v.name: f"{v.pipeline}{'-gpu' if v.gpu else '-cpu'}" for v in RUST_VARIANTS}
    )
    totals["w_trust"] = totals.mean_dtrust * totals.n
    totals["w_recall"] = totals.mean_drecall * totals.n
    grouped = totals.groupby("pipeline").agg(
        n=("n", "sum"), w_trust=("w_trust", "sum"), w_recall=("w_recall", "sum")
    )
    grouped["mean_dtrust"] = grouped.w_trust / grouped.n
    grouped["mean_drecall"] = grouped.w_recall / grouped.n
    return grouped[["n", "mean_dtrust", "mean_drecall"]]


def multi_seed_section(df):
    stats = seed_stats(df)
    multi = sorted(
        {
            (d, c)
            for (d, c, _), r in stats.items()
            if r.n >= MIN_SEEDS_FOR_MEAN_RULE and c != "default"
        }
    )
    lines = [
        "",
        "## Multi-seed configs: trustworthiness mean +- std (recall mean), Rust minus Python mean",
        "",
        "Rule: flagged if Rust mean minus Python mean is below -max(0.01, 2 x pooled std / sqrt(n)), or recall below -0.02.",
        "",
        "| dataset | config | pipeline | Python | Rust | dTrust | dRecall | seeds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dataset, config in multi:
        for variant in RUST_VARIANTS:
            rust = stats.get((dataset, config, variant.name))
            python = stats.get((dataset, config, python_partner(variant)))
            if rust is None or python is None:
                continue
            lines.append(
                f"| {dataset} | `{config}` | {variant.name} | {python.trust_mean:.4f} +- {python.trust_std:.4f} ({python.recall_mean:.3f}) "
                f"| {rust.trust_mean:.4f} +- {rust.trust_std:.4f} ({rust.recall_mean:.3f}) "
                f"| {rust.trust_mean - python.trust_mean:+.4f} | {rust.recall_mean - python.recall_mean:+.4f} | {int(min(rust.n, python.n))} |"
            )
    return lines


def timing_table(df):
    default = df[
        (df.config == "default") & (df.status == "ok") & (df.seed == MAIN_SEED)
    ]
    return default.pivot_table(
        index="variant", columns="dataset", values="time_s", aggfunc="first"
    )


def seed_table(df):
    default = df[(df.config == "default") & (df.status == "ok")]
    grouped = default.groupby(["dataset", "variant"]).agg(
        n=("seed", "count"),
        trust_mean=("trustworthiness", "mean"),
        trust_std=("trustworthiness", "std"),
        recall_std=("knn_recall", "std"),
    )
    return grouped[grouped.n > 1].reset_index()


def error_table(df):
    errors = df[df.status != "ok"]
    return errors[["dataset", "config", "seed", "variant", "reason"]]


def markdown(df, flags, overall, totals):
    n_ok = int((df.status == "ok").sum())
    lines = ["# UMAP sweep: Rust port vs umap-learn 0.6", ""]
    lines.append(
        f"{len(df)} runs ({n_ok} ok, {len(df) - n_ok} error/rejected). Seed {MAIN_SEED} unless noted. "
        "Trustworthiness@15 and kNN recall@15 use a fixed subsample of at most 5000 points. "
        "Deltas are Rust minus the Python run of the same pipeline."
    )
    lines += [
        "",
        "Flag rules: trustworthiness worse by more than max(0.01, 3x seed std); recall worse by more than 0.02; "
        "NaN/inf; collapsed (q95-q5 below 1e-3 on any axis); Rust GPU worse than Rust CPU by the same limits.",
        "",
    ]

    lines += ["## Flagged configs", ""]
    if flags.empty:
        lines += ["None.", ""]
    else:
        lines += [
            "| kind | dataset | config | variant | detail |",
            "|---|---|---|---|---|",
        ]
        for f in flags.itertuples():
            lines.append(
                f"| {f.kind} | {f.dataset} | `{f.config}` | {f.variant} | {f.detail} |"
            )
        lines.append("")

    lines += [
        "## Overall Rust minus Python, per pipeline",
        "",
        "| Rust variant | runs | mean dTrust | mean dRecall |",
        "|---|---|---|---|",
    ]
    for pipeline, row in totals.iterrows():
        lines.append(
            f"| {pipeline} | {int(row.n)} | {row.mean_dtrust:+.4f} | {row.mean_drecall:+.4f} |"
        )

    lines += [
        "",
        "## Per dataset and Rust variant",
        "",
        "| dataset | variant | configs | mean dTrust | worst dTrust | mean dRecall | worst dRecall | median Procrustes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in overall.itertuples():
        lines.append(
            f"| {r.dataset} | {r.variant} | {r.n} | {r.mean_dtrust:+.4f} | {r.min_dtrust:+.4f} | "
            f"{r.mean_drecall:+.4f} | {r.min_drecall:+.4f} | {r.median_procrustes:.3f} |"
        )

    lines += multi_seed_section(df)
    lines += [
        "",
        "## Seed noise (default config)",
        "",
        "| dataset | variant | seeds | trust mean | trust std | recall std |",
        "|---|---|---|---|---|---|",
    ]
    for r in seed_table(df).itertuples():
        lines.append(
            f"| {r.dataset} | {r.variant} | {r.n} | {r.trust_mean:.4f} | {r.trust_std:.4f} | {r.recall_std:.4f} |"
        )

    lines += [
        "",
        "## Wall time (s), default config",
        "",
        "Python seeded runs are single-threaded (umap-learn disables parallelism with random_state); "
        "`_unseeded` rows are multi-threaded. Python seeded timings can include numba compile for new "
        "metric or dimension combinations.",
        "",
    ]
    timing = timing_table(df)
    lines += [
        "| variant | " + " | ".join(timing.columns) + " |",
        "|---|" + "---|" * len(timing.columns),
    ]
    for variant, row in timing.iterrows():
        lines.append(
            f"| {variant} | "
            + " | ".join("" if pd.isna(v) else f"{v:.2f}" for v in row)
            + " |"
        )

    errors = error_table(df)
    lines += ["", f"## Errors and rejected combinations ({len(errors)})", ""]
    if len(errors):
        lines += [
            "| dataset | config | seed | variant | reason |",
            "|---|---|---|---|---|",
        ]
        for e in errors.itertuples():
            lines.append(
                f"| {e.dataset} | `{e.config}` | {e.seed} | {e.variant} | {str(e.reason)[:200]} |"
            )

    lines += [
        "",
        "## All configs: dTrust/dRecall per Rust variant (Rust minus Python)",
        "",
        "| dataset | config | " + " | ".join(v.name for v in RUST_VARIANTS) + " |",
        "|---|---|" + "---|" * len(RUST_VARIANTS),
    ]
    for dataset, config, cells in delta_table(df, flags):
        lines.append(f"| {dataset} | `{config}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_reports():
    df = load_results()
    flags = find_flags(df)
    overall = overall_table(df)
    totals = pipeline_totals(overall)
    SUMMARY_PATH.write_text(markdown(df, flags, overall, totals))
    print(f"Wrote {SUMMARY_PATH}")
