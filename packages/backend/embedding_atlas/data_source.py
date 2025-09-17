# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import json
import os
import zipfile
from io import BytesIO

import pandas as pd

from .utils import cache_path, to_parquet_bytes


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

    def make_archive(self, static_path: str):
        io = BytesIO()
        with zipfile.ZipFile(io, "w", zipfile.ZIP_DEFLATED) as zip:
            zip.writestr(
                "data/metadata.json",
                json.dumps(
                    self.metadata
                    | {"isStatic": True, "database": {"type": "wasm", "load": True}}
                ),
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
