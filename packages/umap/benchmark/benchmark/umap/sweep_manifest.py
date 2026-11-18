"""Build the in-memory manifest of the sweep that the standalone report is generated from."""

import pandas as pd

from benchmark.umap.sweep import embedding_path
from benchmark.umap.sweep_config import (
    COUPLED,
    DATASETS,
    DEFAULT_PARAMS,
    MAIN_SEED,
    ONE_AT_A_TIME,
    PARALLEL_PYTHON_VARIANTS,
    VARIANTS,
    python_partner,
)
from benchmark.umap.sweep_data import SWEEP_DIR, load_dataset


def _relative(path):
    return str(path.relative_to(SWEEP_DIR))


def _clean(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def dataset_entries():
    entries = []
    for spec in DATASETS:
        directory, data, _ = load_dataset(spec.source)
        entries.append(
            {
                "name": spec.name,
                "n": int(data.shape[0]),
                "labels_file": _relative(directory / "labels.bin"),
            }
        )
    return entries


def axis_entries():
    return [
        {"param": "metric", "values": values, "default": "per dataset", "coupled": {}}
        if param == "metric"
        else {
            "param": param,
            "values": values,
            "default": DEFAULT_PARAMS[param],
            "coupled": COUPLED.get(param, {}),
        }
        for param, values in ONE_AT_A_TIME
    ]


def variant_entries():
    entries = []
    for v in [*VARIANTS, *PARALLEL_PYTHON_VARIANTS]:
        entries.append(
            {
                "id": v.name,
                "implementation": v.implementation,
                "pipeline": {
                    "sgd05": "0.5 sgd",
                    "adam": "0.6 adam",
                    "momentum": "0.6 momentum",
                }[v.pipeline],
                "gpu": v.gpu,
                "python_partner": python_partner(v)
                if v.implementation == "rust"
                else None,
                "timing_only": v in PARALLEL_PYTHON_VARIANTS,
            }
        )
    return entries


def axis_of(params):
    """Name the single axis that differs from the defaults (spread wins over its coupled min_dist)."""
    differing = [key for key in DEFAULT_PARAMS if params[key] != DEFAULT_PARAMS[key]]
    if params["metric"] != params["_default_metric"]:
        differing.append("metric")
    if "spread" in differing:
        return "spread"
    return differing[0] if differing else None


def run_entries(df, flags):
    flag_map = {}
    for f in flags.itertuples():
        flag_map.setdefault((f.dataset, f.config, f.variant), []).append(f.kind)
    default_metric = {d.name: d.metric for d in DATASETS}
    runs = []
    for r in df.itertuples():
        params = {key: _clean(getattr(r, key)) for key in [*DEFAULT_PARAMS, "metric"]}
        axis = axis_of({**params, "_default_metric": default_metric[r.dataset]})
        ok = r.status == "ok"
        runs.append(
            {
                "dataset": r.dataset,
                "axis": axis,
                "params": params,
                "seed": int(r.seed),
                "variant": r.variant,
                "status": r.status,
                "reason": _clean(r.reason),
                "file": _relative(
                    embedding_path(r.dataset, r.config, r.seed, r.variant)
                )
                if ok
                else None,
                "n_components": int(r.n_components),
                "time_ms": _clean(r.time_s) and round(r.time_s * 1000, 1),
                "knn_ms": _clean(r.knn_s) and round(r.knn_s * 1000, 1),
                "layout_ms": _clean(r.layout_s) and round(r.layout_s * 1000, 1),
                "trustworthiness": _clean(r.trustworthiness),
                "knn_recall": _clean(r.knn_recall),
                "procrustes_vs_python": _clean(r.procrustes),
                "has_nan": bool(ok and not r.finite),
                "flags": flag_map.get((r.dataset, r.config, r.variant), []),
            }
        )
    return runs


def build_manifest():
    from benchmark.umap.sweep_report import find_flags, load_results

    df = load_results()
    df["finite"] = df["finite"].astype(str).str.lower().eq("true")
    flags = find_flags(df)
    return {
        "main_seed": MAIN_SEED,
        "datasets": dataset_entries(),
        "axes": axis_entries(),
        "variants": variant_entries(),
        "runs": run_entries(df, flags),
    }
