# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import pytest
from embedding_atlas.options import make_embedding_atlas_props


def test_labels_automatic_keeps_default_labels():
    props = make_embedding_atlas_props(labels="automatic")

    assert "embeddingViewLabels" not in props
    assert "embeddingViewConfig" not in props


def test_labels_disabled_turns_off_auto_labels():
    props = make_embedding_atlas_props(labels="disabled")

    assert "embeddingViewLabels" not in props
    assert props["embeddingViewConfig"]["autoLabelEnabled"] is False


def test_labels_list_is_passed_through():
    labels = [{"x": 1.0, "y": 2.0, "text": "a"}]

    props = make_embedding_atlas_props(labels=labels, stop_words=["the"])

    assert props["embeddingViewLabels"] == labels
    assert props["embeddingViewConfig"] == {"autoLabelStopWords": ["the"]}


def test_labels_unknown_string_raises():
    with pytest.raises(ValueError, match="labels must be"):
        make_embedding_atlas_props(labels="auto")
