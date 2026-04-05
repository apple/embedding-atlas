# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrameT

from .cache import async_file_cache_value
from .embedding import create_embedder
from .utils import logger

DEFAULT_MAX_CONCURRENCY = 8


def compute_projection(
    data_frame: IntoDataFrameT,
    *,
    inputs: str,
    modality: str = "auto",
    x: str = "projection_x",
    y: str = "projection_y",
    neighbors: str | None = "neighbors",
    embedder: str | Callable | None = None,
    model: str | None = None,
    batch_size: int | None = None,
    max_concurrency: int | None = None,
    embedder_args: dict | None = None,
    umap_args: dict | None = None,
    cache_root: str | Path | None = None,
) -> IntoDataFrameT:
    """
    Compute embeddings and generate 2D projections for a DataFrame column.

    This is a unified entry point that auto-detects the modality of the input
    data (text, image, audio, or vector) and delegates to the appropriate
    projection function.

    Note: This function cannot be called from within a running async event loop
    (e.g. Jupyter notebooks). Use ``async_compute_projection`` instead.

    Args:
        data_frame: DataFrame containing the data to process. Accepts any
            narwhals-compatible frame (pandas, Polars, cuDF, Modin, etc.).
        inputs: str, column name containing the data to embed/project.
        modality: str, the type of data in the inputs column. One of
            'text', 'image', 'audio', 'vector', or 'auto' (auto-detect).
        x: str, column name where the UMAP X coordinates will be stored.
        y: str, column name where the UMAP Y coordinates will be stored.
        neighbors: str or None, column name where nearest neighbor indices
            will be stored. Set to None to skip.
        embedder: the embedding backend to use. Can be:
            - A string: 'sentence-transformers', 'transformers', or 'litellm'
              to use a built-in embedder.
            - An async callable with signature
              ``async def(batch: list[Any], *, model, embedder_args) -> np.ndarray``
              to use a custom embedder. The function receives a list of canonical
              items (strings for text, ``{"bytes": bytes}`` dicts for image/audio)
              and must return an ndarray of shape ``(batch_size, embedding_dim)``.
            - None (default): auto-selects 'sentence-transformers' for text,
              'transformers' for image/audio.
        model: str or None, name of the embedding model.
        batch_size: int or None, batch size for processing.
        max_concurrency: int or None, maximum number of concurrent batches.
        embedder_args: dict, embedder-specific arguments (e.g., api_key, api_base).
        umap_args: dict, arguments for the UMAP algorithm.
        cache_root: str or Path or None, root directory for caching results.

    Returns:
        A new DataFrame (same type as input) with projection columns added.
    """
    return asyncio.run(
        async_compute_projection(
            data_frame,
            inputs=inputs,
            modality=modality,
            x=x,
            y=y,
            neighbors=neighbors,
            embedder=embedder,
            model=model,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            embedder_args=embedder_args,
            umap_args=umap_args,
            cache_root=cache_root,
        )
    )


async def async_compute_projection(
    data_frame: IntoDataFrameT,
    *,
    inputs: str,
    modality: str = "auto",
    x: str = "projection_x",
    y: str = "projection_y",
    neighbors: str | None = "neighbors",
    embedder: str | Callable | None = None,
    model: str | None = None,
    batch_size: int | None = None,
    max_concurrency: int | None = None,
    embedder_args: dict | None = None,
    umap_args: dict | None = None,
    cache_root: str | Path | None = None,
) -> IntoDataFrameT:
    """
    Async version of ``compute_projection``.

    Use this when calling from within a running async event loop (e.g. Jupyter
    notebooks)::

        df = await async_compute_projection(df, inputs="text")

    See ``compute_projection`` for full argument documentation.
    """
    nw_frame = nw.from_native(data_frame, eager_only=True)
    series = nw_frame[inputs]
    embedder_args = embedder_args or {}
    umap_args = umap_args or {}

    # 1. Infer modality
    if modality == "auto":
        modality = _infer_modality(series)
        logger.info("Auto-detected modality: %s", modality)

    # 2. Convert inputs to canonical format
    if modality == "text":
        canonical = _to_canonical_text(series)
    elif modality in ("image", "audio"):
        canonical = _to_canonical_binary(series)
    elif modality == "vector":
        canonical = _to_canonical_vector(series)
    else:
        raise ValueError(
            f"Unknown modality: {modality}. Must be one of: text, image, audio, vector, auto"
        )

    # 3. Resolve embedder (not needed for vector modality)
    embedder_max_concurrency: int | None = None
    if modality == "vector":
        embedder_name = None
    else:
        if callable(embedder):
            embedder_name = getattr(embedder, "__name__", type(embedder).__name__)
        else:
            if embedder is None:
                embedder = _default_embedder(modality)
            elif embedder == "sentence_transformers":
                embedder = "sentence-transformers"

            if embedder in ("sentence-transformers", "transformers"):
                embedder_max_concurrency = 1

            embedder_name = embedder

    if max_concurrency is None:
        max_concurrency = DEFAULT_MAX_CONCURRENCY
    if embedder_max_concurrency is not None:
        max_concurrency = min(embedder_max_concurrency, max_concurrency)

    cache_key = {
        "version": 1,
        "inputs": canonical,
        "modality": modality,
        "embedder": embedder_name,
        "model": model,
        "umap_args": umap_args,
        "embedder_args": _caching_embedder_args(embedder_args),
    }

    async def run() -> Projection:
        if modality == "vector":
            embedding = np.array(canonical).astype(np.float32)
        else:
            if callable(embedder):
                embed_fn = embedder
            elif embedder is not None:
                embed_fn = create_embedder(
                    embedder,
                    modality=modality,
                    model=model,
                    embedder_args=embedder_args,
                )
            else:
                raise RuntimeError("unreachable")
            embedding = await _run_embedding(
                embed_fn,
                canonical,
                model=model,
                embedder_args=embedder_args,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
            )

        return _run_umap(embedding, umap_args=umap_args)

    proj = await async_file_cache_value(
        cache_key,
        run,
        scope="compute_projection",
        serializer=Projection.serialize,
        deserializer=Projection.deserialize,
        callback=lambda cache_path: print(
            "Using cached projection from " + str(cache_path)
        ),
        cache_root=cache_root,
    )

    # Create a new data frame with the columns from the original, and add proj columns to it.
    backend = nw.get_native_namespace(nw_frame)
    new_columns = [
        nw.new_series(x, proj.projection[:, 0].tolist(), nw.Float64, backend=backend),
        nw.new_series(y, proj.projection[:, 1].tolist(), nw.Float64, backend=backend),
    ]
    if neighbors is not None:
        new_columns.append(
            nw.new_series(
                neighbors,
                [
                    {"distances": b, "ids": a}
                    for a, b in zip(proj.knn_indices, proj.knn_distances)
                ],
                backend=backend,
            )
        )
    return nw.to_native(nw_frame.with_columns(new_columns))


def _detect_binary_modality(data: bytes) -> str:
    """Detect whether binary data is an image or audio based on magic bytes."""
    # Image formats
    if data[:8] == b"\x89PNG\r\n\x1a\n":  # PNG
        return "image"
    if data[:2] == b"\xff\xd8":  # JPEG
        return "image"
    if data[:4] == b"GIF8":  # GIF87a / GIF89a
        return "image"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":  # WebP
        return "image"
    if data[:4] == b"\x00\x00\x01\x00":  # ICO
        return "image"
    if data[:2] in (b"BM",):  # BMP
        return "image"
    if data[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):  # TIFF
        return "image"

    # Audio formats
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":  # WAV
        return "audio"
    if data[:4] == b"fLaC":  # FLAC
        return "audio"
    if data[:4] == b"OggS":  # OGG (Vorbis/Opus)
        return "audio"
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":  # MP3 (ID3 tag or sync frame)
        return "audio"
    if len(data) >= 12 and data[4:8] == b"ftyp":  # MP4/M4A container
        return "audio"
    if data[:4] == b".snd":  # AU
        return "audio"
    if data[:4] in (b"FORM",) and data[8:12] == b"AIFF":  # AIFF
        return "audio"

    # Default to image for unrecognized binary data
    return "image"


def _infer_modality(series: nw.Series) -> str:
    """Infer the modality by inspecting the first non-null value in the series."""
    non_null = series.drop_nulls()
    if len(non_null) == 0:
        return "text"
    sample = non_null[0]

    # Check for vector: list[float] or 1-dimensional ndarray
    if isinstance(sample, np.ndarray) and sample.ndim == 1:
        return "vector"
    if (
        isinstance(sample, list)
        and len(sample) > 0
        and isinstance(sample[0], (int, float))
    ):
        return "vector"

    # Check for image/audio: bytes or {"bytes": ...}
    if isinstance(sample, bytes):
        return _detect_binary_modality(sample)
    if isinstance(sample, dict) and "bytes" in sample:
        raw = sample["bytes"]
        if isinstance(raw, list):
            raw = bytes(raw)
        return _detect_binary_modality(raw)

    # Default to text
    return "text"


def _to_canonical_text(series: nw.Series) -> list[str]:
    """Convert series to canonical text format: list[str] with nulls as 'null'."""
    return series.fill_null("null").cast(nw.String).to_list()


def _to_canonical_binary(series: nw.Series) -> list[dict]:
    """Convert series to canonical image format: list[{"bytes": bytes}]."""
    result = []
    for value in series.to_list():
        if isinstance(value, bytes):
            result.append({"bytes": value})
        elif isinstance(value, dict) and "bytes" in value:
            raw = value["bytes"]
            if isinstance(raw, list):
                raw = bytes(raw)
            result.append({"bytes": raw})
        else:
            raise ValueError(
                f"Cannot convert value of type {type(value)} to image/audio format"
            )
    return result


def _to_canonical_vector(series: nw.Series) -> list[np.ndarray]:
    """Convert series to canonical vector format: list[ndarray[float32]]."""
    result = []
    for value in series.to_list():
        if isinstance(value, np.ndarray):
            result.append(value.astype(np.float32))
        else:
            result.append(np.array(value, dtype=np.float32))
    return result


@dataclass
class Projection:
    # Array with shape (N, embedding_dim), the high-dimensional embedding
    projection: np.ndarray

    knn_indices: np.ndarray
    knn_distances: np.ndarray

    @staticmethod
    def serialize(value: "Projection", fd: IO[bytes]) -> None:
        np.savez(
            fd,
            projection=value.projection,
            knn_indices=value.knn_indices,
            knn_distances=value.knn_distances,
        )

    @staticmethod
    def deserialize(fd: IO[bytes]) -> "Projection":
        d = np.load(fd, allow_pickle=False)
        return Projection(
            projection=d["projection"],
            knn_indices=d["knn_indices"],
            knn_distances=d["knn_distances"],
        )


def _run_umap(
    hidden_vectors: np.ndarray,
    *,
    umap_args: dict | None = None,
) -> Projection:
    if umap_args is None:
        umap_args = {}

    logger.info("Running UMAP for input with shape %s...", str(hidden_vectors.shape))  # type: ignore

    import umap
    from umap.umap_ import nearest_neighbors

    metric = umap_args.get("metric", "cosine")
    n_neighbors = umap_args.get("n_neighbors", 15)

    knn = nearest_neighbors(
        hidden_vectors,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds=None,
        angular=False,
        random_state=umap_args.get("random_state"),
    )

    kwargs = {k: v for k, v in umap_args.items() if k != "metric"}
    proj = umap.UMAP(**kwargs, precomputed_knn=knn, metric=metric)
    result: np.ndarray = proj.fit_transform(hidden_vectors)  # type: ignore

    return Projection(projection=result, knn_indices=knn[0], knn_distances=knn[1])


async def _run_embedding(
    fn: Callable,
    data: list,
    *,
    model: str | None,
    embedder_args: dict,
    batch_size: int | None,
    max_concurrency: int | None,
) -> np.ndarray:
    """Run an embedder function over *data* in batches, return concatenated result."""
    batch_size = batch_size or 32
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    logger.info(
        "Running embedding for %d items in %d batches (batch_size=%d)...",
        len(data),
        len(batches),
        batch_size,
    )

    from .async_map import async_map

    results = await async_map(
        batches,
        lambda b: fn(b, model=model, embedder_args=embedder_args),
        concurrency=max_concurrency or 1,
        max_retry=10,
        description="Embedding",
    )
    return np.concatenate(results, axis=0)


def _default_embedder(modality: str):
    if modality == "text":
        return "sentence-transformers"
    else:
        return "transformers"


def _caching_embedder_args(embedder_args: dict) -> dict:
    IGNORED_KEYS = ["api_key", "api_base"]
    return {
        key: value for key, value in embedder_args.items() if key not in IGNORED_KEYS
    }
