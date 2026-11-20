"""Standalone single-file HTML report of the sweep: no server, no network, no build step."""

import base64
import json
import os
from pathlib import Path

import numpy as np

from benchmark.umap.sweep_data import SWEEP_DIR

REPORT_DIR = Path(__file__).parent / "report"
DEFAULT_POINTS = 1500
SUBSET_SEED = 0
LIB_PLACEHOLDER = "/*__REPORT_LIB__*/"
DATA_PLACEHOLDER = "__REPORT_DATA__"
RUN_FIELDS = (
    "dataset variant seed axis params status reason time_ms knn_ms layout_ms "
    "trustworthiness knn_recall procrustes_vs_python has_nan flags"
).split()


def _base64(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def _round_float(value):
    return float(f"{value:.8g}")


def quantize_uint16(values):
    """Map values to uint16 codes between their finite min and max. Non-finite values go to the middle."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype="<u2"), 0.0, 0.0
    low = _round_float(values[finite].min())
    high = _round_float(values[finite].max())
    filled = np.where(finite, values, (low + high) / 2)
    if high <= low:
        return np.zeros(len(values), dtype="<u2"), low, high
    codes = np.floor((filled - low) / (high - low) * 65535 + 0.5)
    return np.clip(codes, 0, 65535).astype("<u2"), low, high


def encode_run_points(embedding):
    """First two dimensions as all x codes then all y codes, plus the (xMin, xMax, yMin, yMax) ranges."""
    x_codes, x_low, x_high = quantize_uint16(embedding[:, 0])
    y_codes, y_low, y_high = quantize_uint16(embedding[:, 1])
    return {
        "q": _base64(np.concatenate([x_codes, y_codes]), "<u2"),
        "r": [x_low, x_high, y_low, y_high],
    }


def choose_subset(n, count):
    """One fixed, sorted random subset of row indices. The same n always gives the same subset."""
    if count >= n:
        return np.arange(n)
    return np.sort(
        np.random.RandomState(SUBSET_SEED).choice(n, size=count, replace=False)
    )


def dataset_subsets(manifest, sweep_dir, points):
    subsets = {}
    for entry in manifest["datasets"]:
        labels = np.fromfile(sweep_dir / entry["labels_file"], dtype=np.uint8)
        indices = choose_subset(entry["n"], points)
        subsets[entry["name"]] = {
            "indices": indices,
            "entry": {
                "name": entry["name"],
                "n": entry["n"],
                "count": len(indices),
                "indices": _base64(indices, "<u4"),
                "labels": _base64(labels[indices], "<u1"),
            },
        }
    return subsets


def read_embedding(sweep_dir, run, n):
    flat = np.fromfile(sweep_dir / run["file"], dtype="<f4")
    embedding = flat.reshape(-1, run["n_components"])
    if len(embedding) != n:
        raise ValueError(f"{run['file']} has {len(embedding)} rows, expected {n}")
    return embedding


def build_report_data(manifest, sweep_dir, points):
    subsets = dataset_subsets(manifest, sweep_dir, points)
    sizes = {entry["name"]: entry["n"] for entry in manifest["datasets"]}
    runs = []
    for run in manifest["runs"]:
        slim = {key: run[key] for key in RUN_FIELDS}
        if run["file"] is not None:
            embedding = read_embedding(sweep_dir, run, sizes[run["dataset"]])
            slim.update(
                encode_run_points(embedding[subsets[run["dataset"]]["indices"]])
            )
        runs.append(slim)
    return {
        "main_seed": manifest["main_seed"],
        "datasets": [subset["entry"] for subset in subsets.values()],
        "axes": manifest["axes"],
        "variants": manifest["variants"],
        "runs": runs,
    }


def serialize_data(data):
    """Compact JSON that is safe inside a script element."""
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    return text.replace("<", "\\u003c")


def render_html(data):
    template = (REPORT_DIR / "report.html").read_text()
    library = (REPORT_DIR / "report_lib.js").read_text()
    return template.replace(LIB_PLACEHOLDER, library).replace(
        DATA_PLACEHOLDER, serialize_data(data)
    )


def write_report_html(
    manifest, points=DEFAULT_POINTS, sweep_dir=SWEEP_DIR, output=None
):
    output = Path(output) if output else sweep_dir / "report.html"
    html = render_html(build_report_data(manifest, Path(sweep_dir), points))
    temp = output.with_suffix(".html.tmp")
    temp.write_text(html)
    os.replace(temp, output)
    print(
        f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {points} points per embedding)"
    )
    return output
