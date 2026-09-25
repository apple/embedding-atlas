# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Unit tests for the litellm embedder (no network access)."""

import io

import numpy as np
import pytest
from embedding_atlas.embedding import create_embedder
from embedding_atlas.projection import _detect_mime_type
from PIL import Image


def _image_bytes(fmt: str) -> bytes:
    img = Image.new("RGB", (4, 4), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _detect_mime_type
# ---------------------------------------------------------------------------


class TestDetectMimeType:
    @pytest.mark.parametrize(
        "fmt, mime",
        [
            ("PNG", "image/png"),
            ("JPEG", "image/jpeg"),
            ("GIF", "image/gif"),
            ("WEBP", "image/webp"),
            ("BMP", "image/bmp"),
            ("TIFF", "image/tiff"),
        ],
    )
    def test_pillow_encoded_images(self, fmt, mime):
        assert _detect_mime_type(_image_bytes(fmt)) == mime

    def test_big_endian_tiff(self):
        assert _detect_mime_type(b"MM\x00\x2a" + b"\x00" * 8) == "image/tiff"

    def test_heic_and_avif(self):
        heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"
        avif = b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1miaf"
        assert _detect_mime_type(heic) == "image/heic"
        assert _detect_mime_type(avif) == "image/avif"

    def test_audio(self):
        wav = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 8
        assert _detect_mime_type(wav) == "audio/wav"

    def test_unknown_bytes_return_none(self):
        assert _detect_mime_type(b"\x00\x01\x02\x03" + b"\x00" * 8) is None
        assert _detect_mime_type(b"") is None


# ---------------------------------------------------------------------------
# litellm image embedder
# ---------------------------------------------------------------------------


@pytest.fixture()
def captured_inputs(monkeypatch):
    """Replace litellm.aembedding with a stub that records the inputs it gets."""
    calls: list[list[str]] = []

    async def fake_aembedding(*, input, model, **kwargs):
        calls.append(list(input))

        class _Response:
            data = [{"embedding": [0.0, 1.0, 2.0]} for _ in input]

        return _Response()

    monkeypatch.setattr("litellm.aembedding", fake_aembedding)
    return calls


def _run_image_embedder(batch):
    import asyncio

    embed = create_embedder(
        "litellm", modality="image", model="some/model", embedder_args={}
    )
    return asyncio.run(embed(batch, model="some/model", embedder_args={}))


def test_image_data_url_uses_actual_mime_type(captured_inputs):
    batch = [
        {"bytes": _image_bytes("JPEG")},
        {"bytes": _image_bytes("PNG")},
        {"bytes": _image_bytes("WEBP")},
    ]
    result = _run_image_embedder(batch)

    prefixes = [inputs[0].split(";", 1)[0] for inputs in captured_inputs]
    assert prefixes == ["data:image/jpeg", "data:image/png", "data:image/webp"]
    assert all(
        inputs[0].split(";", 1)[1].startswith("base64,") for inputs in captured_inputs
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
