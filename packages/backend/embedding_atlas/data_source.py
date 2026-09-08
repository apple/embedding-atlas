# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import json
import os
import pathlib
import re
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


_TABLE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validate_additional_tables(additional_tables: dict | None):
    if not additional_tables:
        return
    for name in additional_tables:
        if not _TABLE_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid table name: {name!r}")
        if name == "dataset":
            raise ValueError("table name 'dataset' is reserved")


class DataSource:
    def __init__(
        self,
        identifier: str,
        dataset: "pd.DataFrame | None",
        metadata: dict,
        additional_tables: dict | None = None,
    ):
        """Wraps a dataset plus metadata and a local cache store.

        ``dataset`` may be ``None`` for fast-path server-mode loads that
        keep the data inside a DuckDB connection and never materialize a
        pandas DataFrame. In that case the ``/data/dataset.parquet`` and
        archive-export endpoints lazily rebuild the parquet from DuckDB.
        """
        _validate_additional_tables(additional_tables)
        self.identifier = identifier
        self.dataset = dataset
        self.metadata = metadata
        self.additional_tables = additional_tables or {}
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

    def _build_metadata(
        self,
        metadata_overrides: dict | None = None,
    ) -> dict:
        db_meta: dict = {"type": "wasm", "load": True}
        if self.additional_tables:
            db_meta["additionalTables"] = [
                {"name": name, "url": f"tables/{name}.parquet"}
                for name in self.additional_tables
            ]
        metadata = self.metadata | {
            "isStatic": True,
            "database": db_meta,
        }
        if metadata_overrides:
            metadata = _deep_merge(metadata, metadata_overrides)
        return metadata

    def make_archive(
        self,
        static_path: str,
        metadata_overrides: dict | None = None,
    ):
        io = BytesIO()

        full_parquet = to_parquet_bytes(self.dataset)
        size = len(full_parquet)

        # Split parquet into parts if it exceeds 50MB (GitHub Pages limit is 100MB, but 50MB is safer)
        MAX_SIZE = 50 * 1024 * 1024
        parts = []
        if size > MAX_SIZE:
            num_parts = (size // MAX_SIZE) + 1
            rows_per_part = len(self.dataset) // num_parts + 1
            for i in range(num_parts):
                start = i * rows_per_part
                end = (i + 1) * rows_per_part
                part_df = self.dataset.iloc[start:end]
                if len(part_df) > 0:
                    parts.append(
                        (f"data/dataset_{i}.parquet", to_parquet_bytes(part_df))
                    )
        else:
            parts.append(("data/dataset.parquet", full_parquet))

        file_names = [os.path.basename(p[0]) for p in parts]

        with zipfile.ZipFile(io, "w", zipfile.ZIP_DEFLATED) as zip:
            zip.writestr(
                "data/metadata.json",
                json.dumps(
                    _deep_merge(
                        self._build_metadata(metadata_overrides),
                        {"database": {"files": file_names}},
                    )
                ),
            )
            for path, data in parts:
                zip.writestr(path, data)
            for name, df in self.additional_tables.items():
                zip.writestr(f"data/tables/{name}.parquet", to_parquet_bytes(df))
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
        if self.additional_tables:
            tables_dir = data_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            for name, df in self.additional_tables.items():
                (tables_dir / f"{name}.parquet").write_bytes(to_parquet_bytes(df))

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
