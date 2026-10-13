# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import pandas as pd
from click.testing import CliRunner

from embedding_atlas import cli, projection


def test_device_is_forwarded_in_embedder_args(monkeypatch, tmp_path):
    captured = {}

    def fake_compute_projection(data, **kwargs):
        captured.update(kwargs)
        result = data.copy()
        result[kwargs["x"]] = 0.0
        result[kwargs["y"]] = 0.0
        if kwargs["neighbors"] is not None:
            result[kwargs["neighbors"]] = [{"ids": [], "distances": []}]
        return result

    class FakeDataSource:
        def __init__(self, *args, **kwargs):
            pass

        def export_to_folder(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        cli, "load_datasets", lambda *args, **kwargs: pd.DataFrame({"text": ["hi"]})
    )
    monkeypatch.setattr(projection, "compute_projection", fake_compute_projection)
    monkeypatch.setattr(cli, "DataSource", FakeDataSource)

    result = CliRunner().invoke(
        cli.main,
        [
            "input.parquet",
            "--text",
            "text",
            "--device",
            "cpu",
            "--export-application",
            str(tmp_path / "export"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["embedder_args"] == {"device": "cpu"}
