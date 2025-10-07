# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import inquirer
import numpy as np
import pandas as pd
import pyarrow as pa
from platformdirs import user_cache_path

logger = logging.getLogger()


def load_pandas_data(url: str) -> pd.DataFrame:
    suffix = Path(url).suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(url)
    elif suffix == ".json" or suffix == ".ndjson":
        df = pd.read_json(url)
    elif suffix == ".jsonl":
        df = pd.read_json(url, lines=True)
    else:
        df = pd.read_csv(url)
    return df


def load_huggingface_data(filename: str, splits: list[str] | None) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "⚠️ Loading Hugging Face datasets requires the `datasets` package to be installed. Please run `pip install datasets`, then try again."
        )
        exit(-1)

    ds: Any = load_dataset(filename)

    if splits is None or len(splits) == 0:
        ds_split_options = []
        for key in ds.keys():
            option = (f"{key} ({ds[key].num_rows} rows)", key)
            ds_split_options.append(option)
        split_question = [
            inquirer.Checkbox(
                "split",
                message=f"Select which data splits you want to load for dataset [{filename}]",
                choices=sorted(ds_split_options),
            ),
        ]
        splits = inquirer.prompt(split_question)["split"]  # type: ignore

    if splits is None or len(splits) == 0:
        raise ValueError("must select at least one split")

    dfs = []
    for split in splits:
        df = ds[split].to_pandas()
        df["split"] = split
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def arrow_to_bytes(arrow: pa.Table | pa.RecordBatchReader):
    if isinstance(arrow, pa.Table):
        # DuckDB version < 1.4.0 returns a pa.Table
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, arrow.schema) as writer:
            writer.write(arrow)
        return sink.getvalue().to_pybytes()
    else:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, arrow.schema) as writer:
            for batch in arrow:
                writer.write_batch(batch)
        return sink.getvalue().to_pybytes()


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    class NoCloseBytesIO(BytesIO):
        def close(self):
            pass

        def actually_close(self):
            super().close()

    bytes_io = NoCloseBytesIO()
    df.to_parquet(bytes_io)
    result = bytes_io.getvalue()
    bytes_io.actually_close()
    return result


def cache_path(*subfolders: str, mkdir=True) -> Path:
    p = user_cache_path("embedding_atlas")
    for f in subfolders:
        p = p / f
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p


class Hasher:
    def __init__(self):
        self.hash = hashlib.sha256()
        self.counter = 0

    def _emit(self, type: bytes, data: bytes):
        self.hash.update(type + b"{")
        self.hash.update(data)
        self.hash.update(b"}")

    def _emit_value(self, value):
        if isinstance(value, bytes):
            self._emit(b"bytes", value)
        elif isinstance(value, str):
            self._emit(b"str", value.encode("utf-8"))
        elif isinstance(value, np.ndarray):
            self._emit(b"np.ndarray", value.tobytes())
        elif isinstance(value, list):
            self.hash.update(b"list{")
            for item in value:
                self._emit_value(item)
            self.hash.update(b"}")
        elif isinstance(value, dict):
            self.hash.update(b"dict{")
            for key, value in value.items():
                self._emit_value(key)
                self._emit_value(value)
            self.hash.update(b"}")
        else:
            self._emit(b"json", json.dumps(value, sort_keys=True).encode("utf-8"))

    def update(self, value):
        self._emit_value(value)

    def hexdigest(self):
        return self.hash.hexdigest()
