# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Tests for compute_projection using a placeholder embedder (no real models)."""

import io
import shutil

import numpy as np
import pandas as pd
import polars as pl
import pytest
from embedding_atlas.embedding import create_embedder
from embedding_atlas.projection import compute_projection
from PIL import Image

NUM_SAMPLES = 30
EMBEDDING_DIM = 16


async def _fake_embedder(batch: list, *, model, embedder_args) -> np.ndarray:
    """Return deterministic random vectors seeded by batch length."""
    rng = np.random.RandomState(len(batch))
    return rng.randn(len(batch), EMBEDDING_DIM).astype(np.float32)


def _make_random_image_bytes(width=64, height=64, seed=0) -> bytes:
    rng = np.random.RandomState(seed)
    pixels = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def cache_root(tmp_path):
    path = tmp_path / "cache"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def text_df():
    return pd.DataFrame({"text": [f"sentence {i}" for i in range(NUM_SAMPLES)]})


@pytest.fixture()
def image_df():
    images = [{"bytes": _make_random_image_bytes(seed=i)} for i in range(NUM_SAMPLES)]
    return pd.DataFrame({"image": images})


@pytest.fixture()
def vector_df():
    rng = np.random.RandomState(42)
    vectors = [rng.randn(EMBEDDING_DIM).astype(np.float32) for _ in range(NUM_SAMPLES)]
    return pd.DataFrame({"vec": vectors})


# ---------------------------------------------------------------------------
# Text modality
# ---------------------------------------------------------------------------


def test_text(text_df, cache_root):
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    assert "projection_x" in result.columns
    assert "projection_y" in result.columns
    assert "neighbors" in result.columns
    assert len(result) == NUM_SAMPLES


def test_text_auto_modality(text_df, cache_root):
    result = compute_projection(
        text_df,
        inputs="text",
        modality="auto",
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    assert len(result) == NUM_SAMPLES
    assert result["projection_x"].notna().all()


# ---------------------------------------------------------------------------
# Image modality
# ---------------------------------------------------------------------------


def test_image(image_df, cache_root):
    result = compute_projection(
        image_df,
        inputs="image",
        modality="image",
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    assert len(result) == NUM_SAMPLES
    assert "projection_x" in result.columns


def test_image_auto_modality(image_df, cache_root):
    result = compute_projection(
        image_df,
        inputs="image",
        modality="auto",
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    assert len(result) == NUM_SAMPLES


# ---------------------------------------------------------------------------
# Vector modality (no embedder needed)
# ---------------------------------------------------------------------------


def test_vector(vector_df, cache_root):
    result = compute_projection(
        vector_df,
        inputs="vec",
        modality="vector",
        cache_root=cache_root,
    )
    assert len(result) == NUM_SAMPLES
    assert "projection_x" in result.columns


def test_vector_auto_modality(vector_df, cache_root):
    result = compute_projection(
        vector_df,
        inputs="vec",
        modality="auto",
        cache_root=cache_root,
    )
    assert len(result) == NUM_SAMPLES


# ---------------------------------------------------------------------------
# Custom embedder receives correct data
# ---------------------------------------------------------------------------


def test_custom_embedder_receives_text_strings(text_df, cache_root):
    """Verify the custom embedder receives canonical text (list[str])."""
    received = []

    async def capture_embedder(batch, *, model, embedder_args):
        received.extend(batch)
        rng = np.random.RandomState(0)
        return rng.randn(len(batch), EMBEDDING_DIM).astype(np.float32)

    compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder=capture_embedder,
        cache_root=cache_root,
    )
    assert len(received) == NUM_SAMPLES
    assert all(isinstance(item, str) for item in received)


def test_custom_embedder_receives_image_dicts(image_df, cache_root):
    """Verify the custom embedder receives canonical image dicts."""
    received = []

    async def capture_embedder(batch, *, model, embedder_args):
        received.extend(batch)
        rng = np.random.RandomState(0)
        return rng.randn(len(batch), EMBEDDING_DIM).astype(np.float32)

    compute_projection(
        image_df,
        inputs="image",
        modality="image",
        embedder=capture_embedder,
        cache_root=cache_root,
    )
    assert len(received) == NUM_SAMPLES
    assert all(isinstance(item, dict) and "bytes" in item for item in received)


def test_custom_embedder_receives_model_and_args(text_df, cache_root):
    """Verify model and embedder_args are forwarded to the custom embedder."""
    captured_kwargs = {}

    async def capture_embedder(batch, *, model, embedder_args):
        captured_kwargs["model"] = model
        captured_kwargs["embedder_args"] = embedder_args
        rng = np.random.RandomState(0)
        return rng.randn(len(batch), EMBEDDING_DIM).astype(np.float32)

    compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder=capture_embedder,
        model="my-model",
        embedder_args={"api_key": "test-key"},
        cache_root=cache_root,
    )
    assert captured_kwargs["model"] == "my-model"
    assert captured_kwargs["embedder_args"] == {"api_key": "test-key"}


# ---------------------------------------------------------------------------
# API behavior
# ---------------------------------------------------------------------------


def test_custom_column_names(text_df, cache_root):
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        x="cx",
        y="cy",
        neighbors="nn",
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    assert "cx" in result.columns
    assert "cy" in result.columns
    assert "nn" in result.columns


def test_no_neighbors(text_df, cache_root):
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        neighbors=None,
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    assert "projection_x" in result.columns
    assert "projection_y" in result.columns
    assert "neighbors" not in result.columns


def test_returns_new_dataframe(text_df, cache_root):
    original_columns = list(text_df.columns)
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    assert list(text_df.columns) == original_columns
    assert "projection_x" not in text_df.columns
    assert result is not text_df


def test_preserves_original_data(text_df, cache_root):
    text_df["id"] = range(len(text_df))
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    assert "id" in result.columns
    assert "text" in result.columns
    assert list(result["id"]) == list(range(NUM_SAMPLES))


def test_neighbors_structure(text_df, cache_root):
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    for neighbor in result["neighbors"]:
        assert isinstance(neighbor, dict)
        assert "ids" in neighbor
        assert "distances" in neighbor


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_modality(text_df, cache_root):
    with pytest.raises(ValueError, match="Unknown modality"):
        compute_projection(
            text_df,
            inputs="text",
            modality="unknown",
            embedder=_fake_embedder,
            cache_root=cache_root,
        )


def test_invalid_column(text_df, cache_root):
    with pytest.raises(KeyError):
        compute_projection(
            text_df,
            inputs="nonexistent",
            modality="text",
            embedder=_fake_embedder,
            cache_root=cache_root,
        )


def test_unknown_embedder_name(text_df, cache_root):
    with pytest.raises(ValueError, match="Unknown embedder"):
        compute_projection(
            text_df,
            inputs="text",
            modality="text",
            embedder="nonexistent-engine",
            cache_root=cache_root,
        )


def test_sentence_transformers_rejects_image():
    with pytest.raises(NotImplementedError, match="only supports text"):
        create_embedder(
            "sentence-transformers", modality="image", model=None, embedder_args={}
        )


def test_underscore_embedder_name(text_df, cache_root):
    """Test that 'sentence_transformers' (underscore) is accepted and normalized.

    We can't do a full integration test without a real model, so we verify
    indirectly: pass a vector modality (which skips the embedder entirely)
    with the underscore variant to confirm it doesn't blow up on resolution.
    """
    rng = np.random.RandomState(42)
    vectors = [rng.randn(EMBEDDING_DIM).astype(np.float32) for _ in range(NUM_SAMPLES)]
    df = pd.DataFrame({"vec": vectors})
    result = compute_projection(
        df,
        inputs="vec",
        modality="vector",
        embedder="sentence_transformers",
        cache_root=cache_root,
    )
    assert len(result) == NUM_SAMPLES


# ---------------------------------------------------------------------------
# Polars backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def text_pl():
    return pl.DataFrame({"text": [f"sentence {i}" for i in range(NUM_SAMPLES)]})


@pytest.fixture()
def vector_pl():
    rng = np.random.RandomState(42)
    vectors = [
        rng.randn(EMBEDDING_DIM).astype(np.float32).tolist() for _ in range(NUM_SAMPLES)
    ]
    return pl.DataFrame({"vec": vectors})


def test_text_polars(text_pl, cache_root):
    result = compute_projection(
        text_pl,
        inputs="text",
        modality="text",
        embedder=_fake_embedder,
        cache_root=cache_root,
    )
    assert isinstance(result, pl.DataFrame)
    assert "projection_x" in result.columns
    assert "projection_y" in result.columns
    assert "neighbors" in result.columns
    assert len(result) == NUM_SAMPLES


def test_vector_polars(vector_pl, cache_root):
    result = compute_projection(
        vector_pl,
        inputs="vec",
        modality="vector",
        cache_root=cache_root,
    )
    assert isinstance(result, pl.DataFrame)
    assert "projection_x" in result.columns
    assert len(result) == NUM_SAMPLES
