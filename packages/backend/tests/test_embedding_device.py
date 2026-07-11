# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import asyncio
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from embedding_atlas.embedding import (
    _create_sentence_transformers_embedder,
    _create_transformers_audio_embedder,
    _create_transformers_image_embedder,
    _create_transformers_text_embedder,
)
from embedding_atlas.projection import _caching_embedder_args


def test_device_is_included_in_projection_cache_args():
    assert _caching_embedder_args({"device": "cpu"}) == {"device": "cpu"}


def test_sentence_transformers_forwards_device(monkeypatch):
    sentence_transformer = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=sentence_transformer),
    )

    _create_sentence_transformers_embedder(
        modality="text", model="test-model", embedder_args={"device": "cpu"}
    )

    sentence_transformer.assert_called_once_with(
        "test-model", trust_remote_code=False, device="cpu"
    )


def test_transformers_text_forwards_device(monkeypatch):
    pipeline = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))

    _create_transformers_text_embedder(
        model="test-model", embedder_args={"device": "mps"}
    )

    pipeline.assert_called_once_with(
        "feature-extraction", model="test-model", device="mps"
    )


def test_transformers_image_forwards_device(monkeypatch):
    pipeline = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))

    _create_transformers_image_embedder(
        model="test-model", embedder_args={"device": "cuda:0"}
    )

    pipeline.assert_called_once_with(
        "image-feature-extraction", model="test-model", device="cuda:0"
    )


def test_transformers_audio_moves_model_and_tensors_to_device(monkeypatch):
    model_device = None
    inputs_device = None

    class FakeTensor:
        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return np.array([[1.0, 2.0]], dtype=np.float32)

    class FakeModel:
        def to(self, device):
            nonlocal model_device
            model_device = device
            return self

        def get_audio_features(self, **inputs):
            return FakeTensor()

    class FakeInputs(dict):
        def to(self, device):
            nonlocal inputs_device
            inputs_device = device
            return self

    class FakeProcessor:
        feature_extractor = SimpleNamespace(sampling_rate=16_000)

        def __call__(self, **kwargs):
            return FakeInputs(input_values=kwargs["audio"])

    clap_model = MagicMock()
    clap_model.from_pretrained.return_value = FakeModel()
    clap_processor = MagicMock()
    clap_processor.from_pretrained.return_value = FakeProcessor()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(ClapModel=clap_model, ClapProcessor=clap_processor),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            no_grad=nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(
            read=lambda _: (np.array([0.1, 0.2], dtype=np.float32), 16_000)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "scipy.signal",
        SimpleNamespace(resample=lambda data, _: data),
    )

    embed = _create_transformers_audio_embedder(
        model="test-model",
        embedder_args={"device": "cpu", "trust_remote_code": True},
    )
    result = asyncio.run(embed([{"bytes": b"audio"}], model=None, embedder_args={}))

    clap_model.from_pretrained.assert_called_once_with(
        "test-model", trust_remote_code=True
    )
    assert model_device == "cpu"
    assert inputs_device == "cpu"
    np.testing.assert_array_equal(result, [[1.0, 2.0]])
