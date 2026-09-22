# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import logging
from pathlib import Path
from typing import Any

import inquirer
import narwhals as nw
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from narwhals.typing import IntoDataFrame

logger = logging.getLogger("embedding-atlas")


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
        from datasets import load_dataset, load_from_disk
    except ImportError:
        print(
            "⚠️ Loading Hugging Face datasets requires the `datasets` package to be installed. Please run `pip install datasets`, then try again."
        )
        exit(-1)

    if Path(filename).is_dir():
        ds: Any = load_from_disk(filename)
    else:
        ds: Any = load_dataset(filename)

    if not hasattr(ds, "keys"):
        if splits is not None and len(splits) > 0:
            raise ValueError(
                "Cannot select splits for a single Hugging Face Dataset loaded from disk."
            )
        return ds.to_pandas()

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


def arrow_to_bytes(arrow: pa.RecordBatchReader):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, arrow.schema) as writer:
        for batch in arrow:
            writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def to_parquet_bytes(df: IntoDataFrame) -> bytes:
    arrow_table = nw.from_native(df, eager_only=True).to_arrow()
    sink = pa.BufferOutputStream()
    pq.write_table(arrow_table, sink)
    return sink.getvalue().to_pybytes()


def apply_logging_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: (%(name)s) %(message)s",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_kv(
    text: str,
    allowed_keys: set[str] | None = None,
    allow_empty_values: bool = True,
) -> dict[str, str]:
    """Parse a semicolon-separated `key=value` string into a dict.

    Keys and values are stripped of surrounding whitespace; values may contain
    `=` (only the first `=` of each pair is treated as the separator). Empty
    segments are ignored.

    Raises ValueError on a segment without `=`, on a key outside `allowed_keys`
    (when provided — catches typos like `mainkey`), or on an empty value when
    `allow_empty_values` is False.

    Example: parse_kv("a=1;b=x = y") -> {"a": "1", "b": "x = y"}
    """
    result: dict[str, str] = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid key=value pair: {part!r}")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if allowed_keys is not None and key not in allowed_keys:
            allowed = ", ".join(repr(k) for k in sorted(allowed_keys))
            raise ValueError(f"unknown key {key!r}; allowed keys are {allowed}")
        if not allow_empty_values and value == "":
            raise ValueError(f"empty value for key {key!r}")
        result[key] = value
    return result
