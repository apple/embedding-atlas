import numpy as np
import pandas as pd
from embedding_atlas.projection import compute_vector_projection


def test_compute_vector_projection_basic():
    """Test that compute_vector_projection adds projection columns to the dataframe."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((30, 16)).tolist()
    df = pd.DataFrame({"vec": vectors})

    compute_vector_projection(df, vector="vec")

    assert "projection_x" in df.columns
    assert "projection_y" in df.columns
    assert "neighbors" in df.columns
    assert len(df) == 30
    assert df["projection_x"].notna().all()
    assert df["projection_y"].notna().all()


def test_compute_vector_projection_custom_columns():
    """Test custom column names for x, y, and neighbors."""
    rng = np.random.default_rng(42)
    vectors = [rng.standard_normal(8) for _ in range(30)]
    df = pd.DataFrame({"vec": vectors})

    compute_vector_projection(df, vector="vec", x="cx", y="cy", neighbors="nn")

    assert "cx" in df.columns
    assert "cy" in df.columns
    assert "nn" in df.columns


def test_compute_vector_projection_no_neighbors():
    """Test that neighbors column is not added when neighbors=None."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"vec": rng.standard_normal((30, 8)).tolist()})

    compute_vector_projection(df, vector="vec", neighbors=None)

    assert "projection_x" in df.columns
    assert "projection_y" in df.columns
    assert "neighbors" not in df.columns
