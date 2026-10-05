# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import numpy as np
import pandas as pd


def test_default_row_ids_follow_row_order():
    # Importing the widget needs the built frontend bundle (npm run build).
    from embedding_atlas.widget import EmbeddingAtlasWidget

    # More rows than a single DuckDB row group (122,880).
    n = 200_000
    df = pd.DataFrame({"x": np.zeros(n), "y": np.zeros(n), "pos": np.arange(n)})
    widget = EmbeddingAtlasWidget(df, x="x", y="y")

    rows = widget.selection(format="arrow").read_all()
    assert rows["__row_index__"].to_pylist() == rows["pos"].to_pylist()
