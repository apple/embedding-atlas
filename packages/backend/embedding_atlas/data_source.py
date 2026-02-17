# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import json
import os
import pathlib
import shutil
import zipfile
from io import BytesIO

import pandas as pd

from .utils import cache_path, to_parquet_bytes


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
        self.cache_path = cache_path("cache", self.identifier)

    def cache_set(self, name: str, data):
        path = self.cache_path / name
        with open(path, "w") as f:
            json.dump(data, f)

    def cache_get(self, name: str):
        path = self.cache_path / name
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        else:
            return None

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
            for root, _, files in os.walk(self.cache_path):
                for fn in files:
                    p = os.path.join(
                        "data/cache",
                        os.path.relpath(os.path.join(root, fn), str(self.cache_path)),
                    )
                    zip.write(os.path.join(root, fn), p)
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

        # Copy cache files
        for root, _, files in os.walk(self.cache_path):
            for fn in files:
                src = os.path.join(root, fn)
                rel = os.path.relpath(src, str(self.cache_path))
                dst = data_dir / "cache" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
