"""Datasets, parameter grid, and pipeline variants for the UMAP parameter sweep."""

from dataclasses import dataclass

SEEDS_DEFAULT_CONFIG = [42, 0, 1, 2]
MAIN_SEED = 42
SEEDS_MULTI_CONFIG = [42, 0, 1, 2, 3, 4]
MIN_SEEDS_FOR_MEAN_RULE = 3

DEFAULT_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "spread": 1.0,
    "n_components": 2,
    "repulsion_strength": 1.0,
    "negative_sample_rate": 5,
    "learning_rate": 1.0,
    "n_epochs": None,
    "local_connectivity": 1.0,
    "set_op_mix_ratio": 1.0,
    "init": "auto",
}

ONE_AT_A_TIME = [
    ("n_neighbors", [5, 15, 50, 100]),
    ("min_dist", [0.0, 0.1, 0.5, 0.99]),
    ("spread", [1.0, 2.0]),
    ("metric", ["euclidean", "cosine", "manhattan", "correlation"]),
    ("n_components", [2, 3, 10]),
    ("repulsion_strength", [0.5, 1.0, 4.0, 16.0]),
    ("negative_sample_rate", [1, 5, 15]),
    ("learning_rate", [0.5, 1.0, 2.0]),
    ("n_epochs", [10, 50, 200, 500]),
    ("local_connectivity", [1.0, 2.0]),
    ("set_op_mix_ratio", [0.0, 0.5, 1.0]),
    ("init", ["spectral", "tswspectral", "random", "pca", "recursive"]),
]
# spread is varied together with min_dist 0.5
COUPLED = {"spread": {"min_dist": 0.5}}


@dataclass(frozen=True)
class Variant:
    name: str
    implementation: str  # rust | python
    pipeline: str  # sgd05 | adam | momentum
    gpu: bool = False
    seeded: bool = True


VARIANTS = [
    Variant("rust_05_sgd_cpu", "rust", "sgd05"),
    Variant("rust_05_sgd_gpu", "rust", "sgd05", gpu=True),
    Variant("rust_06_adam_cpu", "rust", "adam"),
    Variant("rust_06_adam_gpu", "rust", "adam", gpu=True),
    Variant("rust_06_momentum_gpu", "rust", "momentum", gpu=True),
    Variant("py_05_compat", "python", "sgd05"),
    Variant("py_06_adam", "python", "adam"),
    Variant("py_06_momentum", "python", "momentum"),
]
# Timing reference: umap-learn without random_state runs multi-threaded.
PARALLEL_PYTHON_VARIANTS = [
    Variant("py_05_compat_unseeded", "python", "sgd05", seeded=False),
    Variant("py_06_adam_unseeded", "python", "adam", seeded=False),
]


def python_partner(variant: Variant) -> str:
    return {
        "sgd05": "py_05_compat",
        "adam": "py_06_adam",
        "momentum": "py_06_momentum",
    }[variant.pipeline]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    metric: str
    source: str  # data directory name
    full_grid: bool
    default_only: bool = False


DATASETS = [
    DatasetSpec("mnist10k_euclidean", "euclidean", "mnist_10000", full_grid=True),
    DatasetSpec("mnist10k_cosine", "cosine", "mnist_10000", full_grid=False),
    DatasetSpec("blobs10k", "euclidean", "blobs_10000_64", full_grid=False),
    DatasetSpec("disconnected5k", "euclidean", "disconnected_5000", full_grid=False),
    DatasetSpec(
        "mnist70k", "euclidean", "mnist_70000", full_grid=False, default_only=True
    ),
]

# Parameters varied on the secondary datasets (the metric is fixed per dataset).
REDUCED_PARAMS = {
    "n_neighbors",
    "min_dist",
    "n_components",
    "n_epochs",
    "init",
    "repulsion_strength",
    "set_op_mix_ratio",
}


def config_id(params: dict, metric: str) -> str:
    diffs = [
        f"{key}={params[key]}"
        for key in DEFAULT_PARAMS
        if params[key] != DEFAULT_PARAMS[key]
    ]
    if params["metric"] != metric:
        diffs.append(f"metric={params['metric']}")
    return ",".join(diffs) or "default"


def configs_for(dataset: DatasetSpec) -> list[dict]:
    """Return the default config followed by the one-at-a-time variations."""
    base = {**DEFAULT_PARAMS, "metric": dataset.metric}
    configs = [base]
    if dataset.default_only:
        return configs
    seen = {config_id(base, dataset.metric)}
    for key, values in ONE_AT_A_TIME:
        if not dataset.full_grid and key not in REDUCED_PARAMS:
            continue
        for value in values:
            params = {**base, **COUPLED.get(key, {}), key: value}
            cid = config_id(params, dataset.metric)
            if cid not in seen:
                seen.add(cid)
                configs.append(params)
    return configs


def is_multi_seed(params: dict) -> bool:
    """Configs where seed noise is large enough to need several seeds."""
    return params["min_dist"] >= 0.5 or params["n_epochs"] == 10
