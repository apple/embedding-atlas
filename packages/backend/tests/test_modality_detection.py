# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Unit tests for modality detection and canonical conversion helpers."""

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from embedding_atlas.projection import (
    _detect_binary_modality,
    _infer_modality,
    _to_canonical_binary,
    _to_canonical_text,
    _to_canonical_vector,
)


def _nw_series(values):
    """Create a narwhals Series from a list of values via pandas."""
    return nw.from_native(pd.Series(values), series_only=True)


# ---------------------------------------------------------------------------
# _detect_binary_modality
# ---------------------------------------------------------------------------


class TestDetectBinaryModality:
    # -- Image formats --

    def test_png(self):
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"

    def test_jpeg(self):
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"

    def test_gif87a(self):
        data = b"GIF87a" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"

    def test_gif89a(self):
        data = b"GIF89a" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"

    def test_webp(self):
        data = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"

    def test_ico(self):
        data = b"\x00\x00\x01\x00" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"

    def test_bmp(self):
        data = b"BM" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"

    def test_tiff_little_endian(self):
        data = b"II\x2a\x00" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"

    def test_tiff_big_endian(self):
        data = b"MM\x00\x2a" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"

    # -- Audio formats --

    def test_wav(self):
        data = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20
        assert _detect_binary_modality(data) == "audio"

    def test_flac(self):
        data = b"fLaC" + b"\x00" * 20
        assert _detect_binary_modality(data) == "audio"

    def test_ogg(self):
        data = b"OggS" + b"\x00" * 20
        assert _detect_binary_modality(data) == "audio"

    def test_mp3_id3(self):
        data = b"ID3\x04\x00" + b"\x00" * 20
        assert _detect_binary_modality(data) == "audio"

    def test_mp3_sync_frame(self):
        data = b"\xff\xfb\x90\x00" + b"\x00" * 20
        assert _detect_binary_modality(data) == "audio"

    def test_mp4_m4a(self):
        data = b"\x00\x00\x00\x1cftyp" + b"M4A " + b"\x00" * 20
        assert _detect_binary_modality(data) == "audio"

    def test_au(self):
        data = b".snd" + b"\x00" * 20
        assert _detect_binary_modality(data) == "audio"

    def test_aiff(self):
        data = b"FORM\x00\x00\x00\x00AIFF" + b"\x00" * 20
        assert _detect_binary_modality(data) == "audio"

    # -- Fallback --

    def test_unrecognized_defaults_to_image(self):
        data = b"\x00\x01\x02\x03" + b"\x00" * 20
        assert _detect_binary_modality(data) == "image"


# ---------------------------------------------------------------------------
# _infer_modality
# ---------------------------------------------------------------------------


class TestInferModality:
    def test_text_strings(self):
        series = _nw_series(["hello", "world"])
        assert _infer_modality(series) == "text"

    def test_text_default_for_all_null(self):
        series = _nw_series([None, None, float("nan")])
        assert _infer_modality(series) == "text"

    def test_text_skips_leading_nulls(self):
        series = _nw_series([None, float("nan"), "hello"])
        assert _infer_modality(series) == "text"

    def test_vector_ndarray(self):
        series = _nw_series([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])])
        assert _infer_modality(series) == "vector"

    def test_vector_list_of_floats(self):
        series = _nw_series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert _infer_modality(series) == "vector"

    def test_vector_list_of_ints(self):
        series = _nw_series([[1, 2, 3], [4, 5, 6]])
        assert _infer_modality(series) == "vector"

    def test_vector_empty_list_falls_through_to_text(self):
        series = _nw_series([[], []])
        assert _infer_modality(series) == "text"

    def test_image_bytes(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        series = _nw_series([png_bytes, png_bytes])
        assert _infer_modality(series) == "image"

    def test_audio_bytes(self):
        wav_bytes = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20
        series = _nw_series([wav_bytes, wav_bytes])
        assert _infer_modality(series) == "audio"

    def test_image_dict_with_bytes(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        series = _nw_series([{"bytes": png_bytes}])
        assert _infer_modality(series) == "image"

    def test_image_dict_with_bytes_as_list(self):
        png_header = list(b"\x89PNG\r\n\x1a\n") + [0] * 20
        series = _nw_series([{"bytes": png_header}])
        assert _infer_modality(series) == "image"

    def test_audio_dict_with_bytes_as_list(self):
        wav_header = list(b"RIFF\x00\x00\x00\x00WAVE") + [0] * 20
        series = _nw_series([{"bytes": wav_header}])
        assert _infer_modality(series) == "audio"


# ---------------------------------------------------------------------------
# _to_canonical_text
# ---------------------------------------------------------------------------


class TestToCanonicalText:
    def test_basic(self):
        series = _nw_series(["hello", "world"])
        result = _to_canonical_text(series)
        assert result == ["hello", "world"]
        assert type(result) is list
        assert all(type(v) is str for v in result)

    def test_null_replaced(self):
        series = _nw_series(["hello", None, float("nan"), "world"])
        result = _to_canonical_text(series)
        assert result == ["hello", "null", "null", "world"]
        assert all(type(v) is str for v in result)

    def test_non_string_cast(self):
        series = _nw_series([1, 2.5, True])
        result = _to_canonical_text(series)
        assert result == ["1", "2.5", "True"]
        assert all(type(v) is str for v in result)


# ---------------------------------------------------------------------------
# _to_canonical_binary
# ---------------------------------------------------------------------------


class TestToCanonicalBinary:
    def test_raw_bytes(self):
        data = b"\x89PNG\r\n\x1a\n"
        result = _to_canonical_binary(_nw_series([data]))
        assert type(result) is list
        assert len(result) == 1
        assert result[0] == {"bytes": data}
        assert type(result[0]) is dict
        assert type(result[0]["bytes"]) is bytes

    def test_dict_with_bytes(self):
        data = b"\x89PNG\r\n\x1a\n"
        result = _to_canonical_binary(_nw_series([{"bytes": data}]))
        assert len(result) == 1
        assert result[0] == {"bytes": data}
        assert type(result[0]) is dict
        assert type(result[0]["bytes"]) is bytes

    def test_dict_with_bytes_as_list(self):
        raw_list = [0x89, 0x50, 0x4E, 0x47]
        result = _to_canonical_binary(_nw_series([{"bytes": raw_list}]))
        assert len(result) == 1
        assert result[0] == {"bytes": bytes(raw_list)}
        assert type(result[0]) is dict
        assert type(result[0]["bytes"]) is bytes

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Cannot convert value of type"):
            _to_canonical_binary(_nw_series([12345]))

    def test_mixed_bytes_and_dicts(self):
        raw = b"\xff\xd8"
        items = [raw, {"bytes": raw}, {"bytes": [0xFF, 0xD8]}]
        result = _to_canonical_binary(_nw_series(items))
        assert type(result) is list
        assert len(result) == 3
        assert all(type(r) is dict and set(r.keys()) == {"bytes"} for r in result)
        assert all(type(r["bytes"]) is bytes for r in result)
        assert result[0]["bytes"] == raw
        assert result[1]["bytes"] == raw
        assert result[2]["bytes"] == bytes([0xFF, 0xD8])


# ---------------------------------------------------------------------------
# _to_canonical_vector
# ---------------------------------------------------------------------------


class TestToCanonicalVector:
    def test_ndarray_input(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = _to_canonical_vector(_nw_series([arr]))
        assert type(result) is list
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float32
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])

    def test_list_input(self):
        result = _to_canonical_vector(_nw_series([[1.0, 2.0, 3.0]]))
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float32
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])

    def test_mixed_ndarray_and_list(self):
        items = [np.array([1.0, 2.0]), [3.0, 4.0]]
        result = _to_canonical_vector(_nw_series(items))
        assert type(result) is list
        assert len(result) == 2
        assert all(isinstance(r, np.ndarray) for r in result)
        assert all(r.dtype == np.float32 for r in result)
