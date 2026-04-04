# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Integration tests for the litellm embedder through compute_projection."""

import io
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from embedding_atlas.projection import compute_projection
from PIL import Image

# Skip unless --run-external is passed AND GEMINI_API_KEY is set.
pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
    ),
]

NUM_SAMPLES = 30
MODEL = "gemini/gemini-embedding-2-preview"


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


def test_text(text_df, cache_root):
    result = compute_projection(
        text_df,
        inputs="text",
        modality="text",
        embedder="litellm",
        model=MODEL,
        cache_root=cache_root,
    )
    _assert_projection_result(result)


def test_text_auto_modality(text_df, cache_root):
    result = compute_projection(
        text_df,
        inputs="text",
        modality="auto",
        embedder="litellm",
        model=MODEL,
        cache_root=cache_root,
    )
    _assert_projection_result(result)


# ---------------------------------------------------------------------------
# Image modality
# ---------------------------------------------------------------------------


def test_image(image_df, cache_root):
    result = compute_projection(
        image_df,
        inputs="image",
        modality="image",
        embedder="litellm",
        model=MODEL,
        cache_root=cache_root,
    )
    _assert_projection_result(result)


def test_image_auto_modality(image_df, cache_root):
    result = compute_projection(
        image_df,
        inputs="image",
        modality="auto",
        embedder="litellm",
        model=MODEL,
        cache_root=cache_root,
    )
    _assert_projection_result(result)
