# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import io
import shutil

import numpy as np
import pandas as pd
import pytest
from embedding_atlas.projection import compute_projection
from PIL import Image

# Skip all tests in this module unless --run-external is passed to pytest.
pytestmark = pytest.mark.external

NUM_SAMPLES = 30
EMBEDDERS = ["sentence-transformers", "transformers"]


def _make_random_image_bytes(width=64, height=64, seed=0) -> bytes:
    """Generate a random RGB image as PNG bytes."""
    rng = np.random.RandomState(seed)
    pixels = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_random_wav_bytes(duration=1.0, sample_rate=48000, seed=0) -> bytes:
    """Generate a random mono WAV file as bytes."""
    import soundfile as sf

    rng = np.random.RandomState(seed)
    samples = rng.randn(int(sample_rate * duration)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


@pytest.fixture()
def cache_root(tmp_path):
    path = tmp_path / "cache"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def text_df():
    """A simple DataFrame with text data."""
    return pd.DataFrame(
        {
            "text": [
                "the cat sat on the mat",
                "the dog chased the ball",
                "a bird flew over the house",
                "fish swim in the ocean",
                "the sun sets behind the mountain",
                "rain falls on the city streets",
                "a child reads a colorful book",
                "music plays softly in the room",
                "the train arrived at the station",
                "flowers bloom in the garden",
                "stars shine brightly at night",
                "the river flows through the valley",
                "a painter works on a canvas",
                "the wind blows through the trees",
                "a chef prepares a delicious meal",
                "snow covers the mountain peaks",
                "the boat sails across the lake",
                "birds sing in the early morning",
                "a student studies for an exam",
                "the moon rises over the horizon",
                "waves crash on the sandy beach",
                "a farmer tends to the crops",
                "the clock ticks on the wall",
                "leaves fall from the old oak tree",
                "a dancer moves across the stage",
                "thunder rumbles in the distance",
                "the library is quiet and peaceful",
                "a runner sprints to the finish line",
                "the cat purrs softly on the couch",
                "a scientist looks through a microscope",
            ]
        }
    )


@pytest.fixture()
def image_df():
    images = [{"bytes": _make_random_image_bytes(seed=i)} for i in range(NUM_SAMPLES)]
    return pd.DataFrame({"image": images})


@pytest.fixture()
def audio_df():
    audios = [{"bytes": _make_random_wav_bytes(seed=i)} for i in range(NUM_SAMPLES)]
    return pd.DataFrame({"audio": audios})


@pytest.fixture()
def vector_df():
    rng = np.random.RandomState(42)
    vectors = [rng.randn(32).astype(np.float32) for _ in range(NUM_SAMPLES)]
    return pd.DataFrame({"vec": vectors})


def _assert_projection_result(result, n=NUM_SAMPLES):
    assert "projection_x" in result.columns
    assert "projection_y" in result.columns
    assert "neighbors" in result.columns
    assert len(result) == n
    assert result["projection_x"].notna().all()
    assert result["projection_y"].notna().all()
    for neighbor in result["neighbors"]:
        assert isinstance(neighbor, dict)
        assert "ids" in neighbor
        assert "distances" in neighbor


# ---------------------------------------------------------------------------
# Text modality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("embedder", EMBEDDERS)
def test_text(text_df, cache_root, embedder):
    """Test text embedding with each embedder."""
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder=embedder,
        cache_root=cache_root,
    )
    _assert_projection_result(result)


@pytest.mark.parametrize("embedder", EMBEDDERS)
def test_text_auto_modality(text_df, cache_root, embedder):
    """Test that auto modality correctly detects text."""
    result = compute_projection(
        text_df,
        inputs="text",
        modality="auto",
        embedder=embedder,
        cache_root=cache_root,
    )
    _assert_projection_result(result)


# ---------------------------------------------------------------------------
# Image modality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("embedder", ["transformers"])
def test_image(image_df, cache_root, embedder):
    """Test image embedding with each embedder."""
    result = compute_projection(
        image_df,
        inputs="image",
        modality="image",
        embedder=embedder,
        cache_root=cache_root,
    )
    _assert_projection_result(result)


def test_image_auto_modality(image_df, cache_root):
    """Test that auto modality correctly detects image."""
    result = compute_projection(
        image_df,
        inputs="image",
        modality="auto",
        cache_root=cache_root,
    )
    _assert_projection_result(result)


# ---------------------------------------------------------------------------
# Audio modality
# ---------------------------------------------------------------------------


def test_audio(audio_df, cache_root):
    """Test audio embedding with CLAP via transformers."""
    result = compute_projection(
        audio_df,
        inputs="audio",
        modality="audio",
        embedder="transformers",
        cache_root=cache_root,
    )
    _assert_projection_result(result)


def test_audio_auto_modality(audio_df, cache_root):
    """Test that auto modality correctly detects audio."""
    result = compute_projection(
        audio_df,
        inputs="audio",
        modality="auto",
        cache_root=cache_root,
    )
    _assert_projection_result(result)


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
    _assert_projection_result(result)


def test_vector_auto_modality(vector_df, cache_root):
    result = compute_projection(
        vector_df,
        inputs="vec",
        modality="auto",
        cache_root=cache_root,
    )
    _assert_projection_result(result)


# ---------------------------------------------------------------------------
# API behavior tests
# ---------------------------------------------------------------------------


def test_custom_column_names(text_df, cache_root):
    """Test custom column names for x, y, and neighbors."""
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        x="cx",
        y="cy",
        neighbors="nn",
        embedder="sentence-transformers",
        cache_root=cache_root,
    )
    assert "cx" in result.columns
    assert "cy" in result.columns
    assert "nn" in result.columns


def test_no_neighbors(text_df, cache_root):
    """Test that neighbors column is not added when neighbors=None."""
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        neighbors=None,
        embedder="sentence-transformers",
        cache_root=cache_root,
    )
    assert "projection_x" in result.columns
    assert "projection_y" in result.columns
    assert "neighbors" not in result.columns


def test_returns_new_dataframe(text_df, cache_root):
    """Test that compute_projection returns a new DataFrame, preserving the original."""
    original_columns = list(text_df.columns)
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder="sentence-transformers",
        cache_root=cache_root,
    )
    assert list(text_df.columns) == original_columns
    assert "projection_x" not in text_df.columns
    assert result is not text_df


def test_preserves_original_data(text_df, cache_root):
    """Test that original DataFrame columns are preserved in the result."""
    text_df["id"] = range(len(text_df))
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder="sentence-transformers",
        cache_root=cache_root,
    )
    assert "id" in result.columns
    assert "text" in result.columns
    assert list(result["id"]) == list(range(30))


def test_neighbors_structure(text_df, cache_root):
    """Test that the neighbors column contains dicts with 'ids' and 'distances' keys."""
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder="sentence-transformers",
        cache_root=cache_root,
    )
    for neighbor in result["neighbors"]:
        assert isinstance(neighbor, dict)
        assert "ids" in neighbor
        assert "distances" in neighbor


def test_invalid_modality(text_df, cache_root):
    """Test that an invalid modality raises ValueError."""
    with pytest.raises(ValueError, match="Unknown modality"):
        compute_projection(
            text_df,
            inputs="text",
            modality="unknown",
            embedder="sentence-transformers",
            cache_root=cache_root,
        )


def test_invalid_column(text_df, cache_root):
    """Test that a missing column raises KeyError."""
    with pytest.raises(KeyError):
        compute_projection(
            text_df,
            inputs="nonexistent",
            modality="text",
            embedder="sentence-transformers",
            cache_root=cache_root,
        )
