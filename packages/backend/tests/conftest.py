# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

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
