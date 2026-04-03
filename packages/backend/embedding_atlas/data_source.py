# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import json
import os
import pathlib
import shutil
import zipfile
from io import BytesIO
from typing import Any

import pandas as pd

from .cache import file_cache_get, file_cache_set
from .utils import to_parquet_bytes


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class DataSource:
    def __init__(
        self,
        identifier: str,
        dataset: pd.DataFrame,
        metadata: dict,
    ):
        self.identifier = identifier
        self.dataset = dataset
        self.metadata = metadata
        self._cache_index: set[str] = set(self._cache_index_load())

    def _cache_index_key(self):
        return [self.identifier, "__index__"]

    def _cache_index_load(self) -> list[str]:
        index = file_cache_get(self._cache_index_key(), scope="DataSource")
        if index is None:
            return []
        return index

    def _cache_index_save(self):
        file_cache_set(
            self._cache_index_key(), sorted(self._cache_index), scope="DataSource"
        )

    def _cache_index_add(self, name: str):
        if name not in self._cache_index:
            self._cache_index.add(name)
            # Re-read from disk and merge to avoid losing entries from other processes
            persisted = set(self._cache_index_load())
            merged = self._cache_index | persisted
            file_cache_set(self._cache_index_key(), sorted(merged), scope="DataSource")

    def cache_set(self, name: str, data):
        file_cache_set([self.identifier, name], data, scope="DataSource")
        self._cache_index_add(name)

    def cache_get(self, name: str):
        return file_cache_get([self.identifier, name], scope="DataSource")

    def cache_items(self) -> dict[str, Any]:
        """Return all cached entries as a dict of {name: value}."""
        result = {}
        for name in self._cache_index:
            value = self.cache_get(name)
            if value is not None:
                result[name] = value
        return result

    def _build_metadata(self, metadata_overrides: dict | None = None) -> dict:
        metadata = self.metadata | {
            "isStatic": True,
            "database": {"type": "wasm", "load": True},
        }
        if metadata_overrides:
            metadata = _deep_merge(metadata, metadata_overrides)
        return metadata

    def make_archive(self, static_path: str, metadata_overrides: dict | None = None):
        io = BytesIO()
        with zipfile.ZipFile(io, "w", zipfile.ZIP_DEFLATED) as zip:
            zip.writestr(
                "data/metadata.json",
                json.dumps(self._build_metadata(metadata_overrides)),
            )
            zip.writestr("data/dataset.parquet", to_parquet_bytes(self.dataset))
            for root, _, files in os.walk(static_path):
                for fn in files:
                    p = os.path.relpath(os.path.join(root, fn), static_path)
                    zip.write(os.path.join(root, fn), p)
            for name, value in self.cache_items().items():
                zip.writestr(
                    f"data/cache/{name}",
                    json.dumps(value),
                )
        return io.getvalue()

    def export_to_folder(
        self,
        static_path: str,
        folder_path: str,
        metadata_overrides: dict | None = None,
    ):
        folder = pathlib.Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        # Write metadata and parquet data
        data_dir = folder / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "metadata.json").write_text(
            json.dumps(self._build_metadata(metadata_overrides))
        )
        (data_dir / "dataset.parquet").write_bytes(to_parquet_bytes(self.dataset))

        # Copy static frontend files
        for root, _, files in os.walk(static_path):
            for fn in files:
                src = os.path.join(root, fn)
                rel = os.path.relpath(src, static_path)
                dst = folder / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        # Write cache files
        cache_dir = data_dir / "cache"
        for name, value in self.cache_items().items():
            cache_file = cache_dir / name
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(value))
