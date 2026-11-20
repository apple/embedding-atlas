"""Resumable UMAP parameter sweep: Rust port versus umap-learn 0.6.

Every finished run appends one row to results/sweep/sweep.csv. A run that is
already in the CSV is skipped, so an interrupted sweep continues where it stopped.
"""

import csv
import hashlib
import json
import subprocess
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes

from benchmark.common import build_rust, rust_binary_path
from benchmark.umap.quality import QualityEvaluator
from benchmark.umap.sweep_config import (
    DATASETS,
    MAIN_SEED,
    PARALLEL_PYTHON_VARIANTS,
    SEEDS_DEFAULT_CONFIG,
    SEEDS_MULTI_CONFIG,
    is_multi_seed,
    VARIANTS,
    Variant,
    config_id,
    configs_for,
    python_partner,
)
from benchmark.umap.sweep_data import SWEEP_DIR, ensure_datasets, load_dataset

CSV_PATH = SWEEP_DIR / "sweep.csv"
EMBEDDINGS_DIR = SWEEP_DIR / "embeddings"
RUST_TIMEOUT_S = 1800

CSV_COLUMNS = [
    "dataset",
    "config",
    "seed",
    "variant",
    "implementation",
    "pipeline",
    "gpu",
    "status",
    "reason",
    "time_s",
    "knn_s",
    "layout_s",
    "trustworthiness",
    "knn_recall",
    "finite",
    "spread_min",
    "spread_max",
    "procrustes",
    "n_neighbors",
    "min_dist",
    "spread",
    "metric",
    "n_components",
    "repulsion_strength",
    "negative_sample_rate",
    "learning_rate",
    "n_epochs",
    "local_connectivity",
    "set_op_mix_ratio",
    "init",
]

RUST_FLAGS = {
    "n_neighbors": "--n-neighbors",
    "min_dist": "--min-dist",
    "spread": "--spread",
    "n_components": "--n-components",
    "repulsion_strength": "--repulsion-strength",
    "negative_sample_rate": "--negative-sample-rate",
    "learning_rate": "--learning-rate",
    "n_epochs": "--n-epochs",
    "local_connectivity": "--local-connectivity",
    "set_op_mix_ratio": "--set-op-mix-ratio",
    "init": "--init",
}
RUST_OPTIMIZER = {"sgd05": "sgd", "adam": "adam", "momentum": "momentum"}
PYTHON_OPTIMIZER = {"sgd05": None, "adam": "adam", "momentum": "momentum"}


def embedding_path(dataset, cid, seed, variant):
    digest = hashlib.md5(cid.encode()).hexdigest()[:10]
    return EMBEDDINGS_DIR / dataset / f"{digest}__{variant}__s{seed}.bin"


def write_embedding(path, embedding):
    np.asarray(embedding, dtype="<f4").tofile(path)


def read_embedding(path, n_components):
    return np.fromfile(path, dtype="<f4").reshape(-1, n_components)


def load_done():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH) as f:
        return {
            (r["dataset"], r["config"], r["seed"], r["variant"])
            for r in csv.DictReader(f)
        }


def append_row(row):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_rust(variant: Variant, data_dir, params, seed, emb_path):
    """Run umap-bench; return (timing dict, stderr text)."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "embedding.bin"
        json_path = Path(tmp) / "result.json"
        cmd = [
            str(rust_binary_path("umap-bench")),
            str(data_dir),
            params["metric"],
            "--compatibility-layout",
            "true" if variant.pipeline == "sgd05" else "false",
            "--optimizer",
            RUST_OPTIMIZER[variant.pipeline],
            "--seed",
            str(seed),
            "--embedding-path",
            str(raw_path),
            "--output",
            str(json_path),
        ]
        if variant.gpu:
            cmd.append("--gpu")
        for key, flag in RUST_FLAGS.items():
            value = params[key]
            if value is not None and not (key == "init" and value == "auto"):
                cmd += [flag, str(value)]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=RUST_TIMEOUT_S, check=False
        )
        if proc.returncode != 0:
            raise RuntimeError(
                proc.stderr.strip().splitlines()[-1] if proc.stderr else "crash"
            )
        result = json.loads(json_path.read_text())
        embedding = np.fromfile(raw_path, dtype="<f4").reshape(
            -1, params["n_components"]
        )
    write_embedding(emb_path, embedding)
    return result, proc.stderr


_warmed = set()


def run_python(variant: Variant, data, params, seed):
    """Run umap-learn; return (embedding, seconds)."""
    import umap as umap_lib

    def build(p, random_state):
        return umap_lib.UMAP(
            n_neighbors=p["n_neighbors"],
            n_components=p["n_components"],
            min_dist=p["min_dist"],
            spread=p["spread"],
            metric=p["metric"],
            n_epochs=p["n_epochs"],
            learning_rate=p["learning_rate"],
            repulsion_strength=p["repulsion_strength"],
            negative_sample_rate=p["negative_sample_rate"],
            local_connectivity=p["local_connectivity"],
            set_op_mix_ratio=p["set_op_mix_ratio"],
            init=p["init"],
            random_state=random_state,
            compatibility_layout=variant.pipeline == "sgd05",
            optimizer=PYTHON_OPTIMIZER[variant.pipeline],
        )

    warm_key = (variant.pipeline, params["metric"], params["n_components"])
    if warm_key not in _warmed:
        _warmed.add(warm_key)
        warm = {**params, "n_neighbors": 10, "n_epochs": 20, "init": "random"}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                build(warm, 0).fit_transform(data[:1500])
        except Exception:
            pass

    reducer = build(params, seed if variant.seeded else None)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = reducer.fit_transform(data)
    return np.asarray(embedding, dtype=np.float32), time.perf_counter() - t0


def embedding_stats(embedding):
    finite = bool(np.isfinite(embedding).all())
    if not finite:
        return {"finite": False, "spread_min": "", "spread_max": ""}
    q5, q95 = np.percentile(embedding, [5, 95], axis=0)
    spread = q95 - q5
    return {
        "finite": True,
        "spread_min": round(float(spread.min()), 4),
        "spread_max": round(float(spread.max()), 4),
    }


def procrustes_disparity(embedding, reference_path):
    n_components = embedding.shape[1]
    if not reference_path.exists():
        return ""
    reference = read_embedding(reference_path, n_components)
    if embedding.shape != reference.shape or not (
        np.isfinite(embedding).all() and np.isfinite(reference).all()
    ):
        return ""
    if embedding.std() == 0 or reference.std() == 0:
        return ""
    return round(float(procrustes(reference, embedding)[2]), 5)


def evaluate_variant(variant, dataset, cid, params, seed, data, data_dir, quality):
    base = {
        "dataset": dataset.name,
        "config": cid,
        "seed": seed,
        "variant": variant.name,
        "implementation": variant.implementation,
        "pipeline": variant.pipeline,
        "gpu": variant.gpu,
        **params,
    }
    emb_path = embedding_path(dataset.name, cid, seed, variant.name)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if variant.implementation == "rust":
            result, stderr = run_rust(variant, data_dir, params, seed, emb_path)
            embedding = read_embedding(emb_path, params["n_components"])
            timing = {
                "time_s": result["time_s"],
                "knn_s": result.get("knn_s", ""),
                "layout_s": result.get("layout_s", ""),
            }
            reason = "fallback reported" if "fall" in stderr.lower() else ""
        else:
            embedding, elapsed = run_python(variant, data, params, seed)
            write_embedding(emb_path, embedding)
            timing = {"time_s": round(elapsed, 3)}
            reason = ""
    except Exception as exc:
        message = " ".join(str(exc).split())[:300]
        if variant.implementation == "python":
            traceback.print_exc()
        return {**base, "status": "error", "reason": f"{type(exc).__name__}: {message}"}

    row = {
        **base,
        "status": "ok",
        "reason": reason,
        **timing,
        **embedding_stats(embedding),
    }
    quality_scores = quality.evaluate(embedding)
    row["trustworthiness"] = quality_scores["trustworthiness"]
    row["knn_recall"] = quality_scores["knn_recall"]
    if variant.implementation == "rust":
        partner = embedding_path(dataset.name, cid, seed, python_partner(variant))
        row["procrustes"] = procrustes_disparity(embedding, partner)
    return row


def write_build_info():
    """Record which Rust sources the umap-bench binary was built from."""
    root = rust_binary_path("umap-bench").parents[2]
    digest = hashlib.sha1()
    for crate in ("umap/umap", "umap/nndescent"):
        for path in sorted((root / "packages" / crate / "src").rglob("*")):
            if path.is_file():
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    info = {
        "git_head": head,
        "rust_sources_sha1": digest.hexdigest(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (SWEEP_DIR / "build_info.json").write_text(json.dumps(info, indent=1))


def run(smoke=False, only_datasets=None, no_gpu=False, skip_build=False):
    ensure_datasets()
    if not skip_build:
        build_rust("umap-bench", gpu=True)
        write_build_info()
    done = load_done()
    variants = [v for v in VARIANTS if not (no_gpu and v.gpu)]
    quality_cache = {}

    for dataset in DATASETS:
        if only_datasets and dataset.name not in only_datasets:
            continue
        data_dir, data, _ = load_dataset(dataset.source)
        configs = configs_for(dataset)[:1] if smoke else configs_for(dataset)
        for index, params in enumerate(configs, 1):
            cid = config_id(params, dataset.metric)
            cache_key = (dataset.name, params["metric"])
            if cache_key not in quality_cache:
                quality_cache[cache_key] = QualityEvaluator(data, params["metric"])
            quality = quality_cache[cache_key]
            is_default = cid == "default"
            seeds = [MAIN_SEED]
            if not smoke and is_default:
                seeds = SEEDS_DEFAULT_CONFIG
            elif not smoke and is_multi_seed(params):
                seeds = SEEDS_MULTI_CONFIG
            for seed in seeds:
                python_first = sorted(
                    variants, key=lambda v: v.implementation != "python"
                )
                extra = (
                    PARALLEL_PYTHON_VARIANTS if is_default and seed == MAIN_SEED else []
                )
                for variant in [*python_first, *extra]:
                    key = (dataset.name, cid, str(seed), variant.name)
                    if key in done:
                        continue
                    print(
                        f"[{dataset.name} {index}/{len(configs)}] {cid} seed={seed} {variant.name}",
                        flush=True,
                    )
                    row = evaluate_variant(
                        variant, dataset, cid, params, seed, data, data_dir, quality
                    )
                    append_row(row)
                    print(
                        f"    {row['status']} time={row.get('time_s', '')} "
                        f"trust={row.get('trustworthiness', '')} "
                        f"recall={row.get('knn_recall', '')} {row['reason']}",
                        flush=True,
                    )
