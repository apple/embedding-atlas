"""Datasets for the parameter sweep: MNIST subsets, gaussian blobs, disconnected blobs."""

import json

import numpy as np
from sklearn.datasets import make_blobs

from benchmark.common import RESULTS_BASE
from benchmark.umap.generate_data import SEED, fetch_mnist, save_dataset
from benchmark.umap.sweep_config import DATASETS

SWEEP_DIR = RESULTS_BASE / "sweep"
DATASETS_DIR = SWEEP_DIR / "datasets"


def _missing(name):
    return not (DATASETS_DIR / name / "meta.json").exists()


def dataset_index(directory=None):
    """Describes the sweep datasets for the browser examples. Paths are relative to index.json."""
    directory = directory or DATASETS_DIR
    entries = []
    for spec in DATASETS:
        folder = directory / spec.source
        meta = json.loads((folder / "meta.json").read_text())
        entries.append(
            {
                "name": spec.name,
                "folder": spec.source,
                "n": meta["n_points"],
                "dim": meta["dim"],
                "metric": spec.metric,
                "labels": (folder / "labels.bin").exists(),
                "bytes": (folder / "data.bin").stat().st_size,
            }
        )
    return {"schema_version": 1, "datasets": entries}


def write_index(directory=None):
    directory = directory or DATASETS_DIR
    path = directory / "index.json"
    path.write_text(json.dumps(dataset_index(directory), indent=2) + "\n")
    return path


def ensure_datasets():
    _create_missing()
    write_index()


def _create_missing():
    if _missing("mnist_70000") or _missing("mnist_10000"):
        data, labels = fetch_mnist()
        save_dataset(data, labels, DATASETS_DIR / "mnist_70000")
        subset = np.sort(
            np.random.RandomState(SEED).choice(len(data), size=10_000, replace=False)
        )
        save_dataset(data[subset], labels[subset], DATASETS_DIR / "mnist_10000")
    if _missing("blobs_10000_64"):
        data, labels = make_blobs(
            n_samples=10_000,
            n_features=64,
            centers=20,
            cluster_std=2.0,
            center_box=(-10, 10),
            random_state=SEED,
        )
        save_dataset(data.astype(np.float32), labels, DATASETS_DIR / "blobs_10000_64")
    if _missing("disconnected_5000"):
        data, labels = make_blobs(
            n_samples=5_000,
            n_features=32,
            centers=5,
            cluster_std=1.0,
            center_box=(-1000, 1000),
            random_state=SEED,
        )
        save_dataset(
            data.astype(np.float32), labels, DATASETS_DIR / "disconnected_5000"
        )


def load_dataset(source):
    directory = DATASETS_DIR / source
    meta = json.loads((directory / "meta.json").read_text())
    data = np.fromfile(directory / "data.bin", dtype="<f4").reshape(
        meta["n_points"], meta["dim"]
    )
    labels = np.fromfile(directory / "labels.bin", dtype=np.uint8)
    return directory, data, labels
