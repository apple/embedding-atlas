# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import os
import sys

# Numba's parallel scheduler can crash UMAP kernels on macOS + CPython 3.12.
# Set this before test modules import UMAP; production guards the affected
# PageRank conversion call directly.
if sys.platform == "darwin" and sys.version_info[:2] == (3, 12):
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="Run tests that require external resources (models, APIs).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-external"):
        return
    skip = pytest.mark.skip(reason="needs --run-external to run")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip)
