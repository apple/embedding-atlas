# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import pytest
from embedding_atlas.utils import load_pandas_data


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_load_pandas_data_newline_delimited_json(tmp_path, suffix):
    path = tmp_path / f"data{suffix}"
    path.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')

    df = load_pandas_data(str(path))

    assert df["a"].tolist() == [1, 2]
    assert df["b"].tolist() == ["x", "y"]
