# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import pandas as pd
import pytest
from embedding_atlas.utils import load_pandas_data


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_load_pandas_data_newline_delimited_json(tmp_path, suffix):
    path = tmp_path / f"data{suffix}"
    path.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')

    df = load_pandas_data(str(path))

    assert df["a"].tolist() == [1, 2]
    assert df["b"].tolist() == ["x", "y"]


@pytest.mark.parametrize(
    "url,reader",
    [
        ("https://example.com/data.parquet?download=true", "read_parquet"),
        ("https://example.com/data.jsonl?X-Amz-Signature=abc", "read_json"),
    ],
)
def test_load_pandas_data_url_with_query_string(monkeypatch, url, reader):
    calls = []
    for name in ["read_csv", "read_json", "read_parquet"]:
        monkeypatch.setattr(
            pd, name, lambda *args, name=name, **kwargs: calls.append((name, args[0]))
        )

    load_pandas_data(url)

    assert calls == [(reader, url)]


@pytest.mark.parametrize(
    "filename,lines",
    [("data.jsonl.gz", True), ("data.ndjson.bz2", True), ("data.json.gz", False)],
)
def test_load_pandas_data_compressed_json(tmp_path, filename, lines):
    path = tmp_path / filename
    expected = pd.DataFrame({"a": [1, 2], "b": ["x, y", "z"]})
    expected.to_json(path, orient="records", lines=lines)

    df = load_pandas_data(str(path))

    assert df.equals(expected)
