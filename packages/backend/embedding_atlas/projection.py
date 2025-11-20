# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from .utils import Hasher, cache_path, logger


@dataclass
class Projection:
    # Array with shape (N, embedding_dim), the high-dimensional embedding
    projection: np.ndarray

    knn_indices: np.ndarray
    knn_distances: np.ndarray

    @staticmethod
    def exists(path: Path):
        return (
            path.with_suffix(".projection.npy").exists()
            and path.with_suffix(".knn_indices.npy").exists()
            and path.with_suffix(".knn_distances.npy").exists()
        )

    @staticmethod
    def save(path: Path, value: "Projection"):
        np.save(
            path.with_suffix(".projection.npy"),
            value.projection,
            allow_pickle=False,
        )
        np.save(
            path.with_suffix(".knn_indices.npy"),
            value.knn_indices,
            allow_pickle=False,
        )
        np.save(
            path.with_suffix(".knn_distances.npy"),
            value.knn_distances,
            allow_pickle=False,
        )

    @staticmethod
    def load(path: Path) -> "Projection":
        return Projection(
            projection=np.load(
                path.with_suffix(".projection.npy"),
                allow_pickle=False,
            ),
            knn_indices=np.load(
                path.with_suffix(".knn_indices.npy"),
                allow_pickle=False,
            ),
            knn_distances=np.load(
                path.with_suffix(".knn_distances.npy"),
                allow_pickle=False,
            ),
        )


def _run_umap(
    hidden_vectors: np.ndarray,
    umap_args: dict = {},
) -> Projection:
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
        random_state=None,
    )

    proj = umap.UMAP(**umap_args, precomputed_knn=knn)
    result: np.ndarray = proj.fit_transform(hidden_vectors)  # type: ignore

    return Projection(projection=result, knn_indices=knn[0], knn_distances=knn[1])


# Signature of a function that takes a list of strings, batch size, model name, and optional args, and returns a numpy array of embeddings.
TextProjectorCallback = Callable[
    [list[str], int, str, dict | None],
    np.ndarray,
]


def _project_text_with_sentence_transformers(
    texts: list[str],
    batch_size: int,
    model: str,
    args: dict | None = None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    default_args = {
        "trust_remote_code": False,
    }
    merged_args = {**default_args, **(args or {})}

    logger.info("Loading model %s...", model)
    transformer = SentenceTransformer(model, **merged_args)
    return transformer.encode(texts, batch_size=batch_size)


def _project_text_with_litellm(
    texts: list[str],
    batch_size: int,
    model: str,
    args: dict | None = None,
) -> np.ndarray:
    from litellm import EmbeddingResponse, aembedding, embedding

    default_args = {
        "sync": False,
    }
    merged_args = {**default_args, **(args or {})}

    def run_sync() -> list[EmbeddingResponse]:
        # Process batches synchronously
        return [
            embedding(
                input=texts[i : i + batch_size],
                model=model,
                **merged_args,
            )
            for i in range(0, len(texts), batch_size)
        ]

    def run_async() -> list[EmbeddingResponse]:
        import asyncio

        async def run_async_coro() -> list[EmbeddingResponse]:
            # Create coroutines for each batch for asynchronous processing
            response_coroutines = [
                aembedding(
                    input=texts[i : i + batch_size],
                    model=model,
                    **merged_args,
                )
                for i in range(0, len(texts), batch_size)
            ]

            return await asyncio.gather(*response_coroutines)

        return asyncio.run(run_async_coro())

    # Determine whether to run synchronously or asynchronously
    should_run_sync: bool = merged_args["sync"]
    responses = run_sync() if should_run_sync else run_async()

    # Unnest embeddings from response wrapper objects
    return np.array(
        [
            np.array(item["embedding"])
            for response in responses
            for item in response.data
        ]
    )


def _projection_for_texts(
    texts: list[str],
    model: str | None = None,
    batch_size: int | None = None,
    text_projector_args: dict | None = None,
    text_projector: TextProjectorCallback | None = None,
    umap_args: dict = {},
) -> Projection:
    if model is None:
        model = "all-MiniLM-L6-v2"
    if text_projector is None:
        text_projector = _project_text_with_sentence_transformers

    # Some arguments may contain sensitive info (e.g., API keys) or do not invalidate the cache, so we exclude them
    excluded_text_projector_args = {"api_key", "api_base", "sync"}
    hashed_text_projector_args = {
        k: v
        for k, v in (text_projector_args or {}).items()
        if k not in excluded_text_projector_args
    }
    hasher = Hasher()
    hasher.update(
        {
            "version": 2,
            "texts": texts,
            "model": model,
            "batch_size": batch_size,
            "text_projector_args": hashed_text_projector_args,
            "text_projector": text_projector.__name__,
            "umap_args": umap_args,
        }
    )
    digest = hasher.hexdigest()
    cpath = cache_path("projections") / digest

    if Projection.exists(cpath):
        logger.info("Using cached projection from %s", str(cpath))
        return Projection.load(cpath)

    # Set default batch size if not provided
    if batch_size is None:
        batch_size = 32
        logger.info(
            "Using default batch size of %d for text. Adjust with --batch-size if you encounter memory issues or want to speed up processing.",
            batch_size,
        )

    logger.info(
        "Running embedding for %d texts with batch size %d using %s...",
        len(texts),
        batch_size,
        text_projector.__name__,
    )
    hidden_vectors = text_projector(texts, batch_size, model, text_projector_args)

    result = _run_umap(hidden_vectors, umap_args)
    Projection.save(cpath, result)
    return result


def _projection_for_images(
    images: list,
    model: str | None = None,
    trust_remote_code: bool = False,
    batch_size: int | None = None,
    umap_args: dict = {},
) -> Projection:
    if model is None:
        model = "google/vit-base-patch16-384"
    hasher = Hasher()
    hasher.update(
        {
            "version": 1,
            "images": images,
            "model": model,
            "batch_size": batch_size,
            "umap_args": umap_args,
        }
    )
    digest = hasher.hexdigest()
    cpath = cache_path("projections") / (digest + ".npy")

    if Projection.exists(cpath):
        logger.info("Using cached projection from %s", str(cpath))
        return Projection.load(cpath)

    # Import on demand.
    from io import BytesIO

    import torch
    import tqdm
    from PIL import Image
    from transformers import pipeline

    def load_image(value):
        if isinstance(value, bytes):
            return Image.open(BytesIO(value)).convert("RGB")
        elif isinstance(value, dict) and "bytes" in value:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        else:
            raise ValueError("invalid image value")

    logger.info("Loading model %s...", model)

    pipe = pipeline("image-feature-extraction", model=model, device_map="auto")

    # Set default batch size if not provided
    if batch_size is None:
        batch_size = 16
        logger.info(
            "Using default batch size of %d for images. Adjust with --batch-size if you encounter memory issues or want to speed up processing.",
            batch_size,
        )

    logger.info(
        "Running embedding for %d images with batch size %d...", len(images), batch_size
    )
    tensors = []

    current_batch = []

    @torch.no_grad()
    def process_batch():
        rs: torch.Tensor = pipe(current_batch, return_tensors=True)  # type: ignore
        current_batch.clear()
        for r in rs:
            if len(r.shape) == 3:
                r = r.mean(1)
            assert len(r.shape) == 2
            tensors.append(r)

    for image in tqdm.tqdm(images, smoothing=0.1):
        current_batch.append(load_image(image))
        if len(current_batch) >= batch_size:
            process_batch()
    process_batch()

    hidden_vectors = torch.concat(tensors).to(torch.float32).cpu().numpy()

    result = _run_umap(hidden_vectors, umap_args)
    Projection.save(cpath, result)
    return result


def _find_text_projector_callback(name: str) -> TextProjectorCallback:
    projector_map = {
        "sentence_transformers": _project_text_with_sentence_transformers,
        "litellm": _project_text_with_litellm,
    }

    if name in projector_map:
        return projector_map[name]

    raise ValueError(
        f"Unknown text projector: {name}. Must be one of: {list(projector_map.keys())}"
    )


def compute_text_projection(
    data_frame: pd.DataFrame,
    text: str,
    x: str = "projection_x",
    y: str = "projection_y",
    neighbors: str | None = "neighbors",
    model: str | None = None,
    batch_size: int | None = None,
    text_projector: Literal[
        "sentence_transformers",
        "litellm",
    ] = "sentence_transformers",
    umap_args: dict = {},
    **kwargs,
):
    """
    Compute text embeddings and generate 2D projections using UMAP.

    This function processes text data by creating embeddings and then reducing the
    dimensionality to 2D coordinates using UMAP for visualization purposes.

    Args:
        data_frame: pandas DataFrame containing the text data to process.
        text: str, column name containing the texts to embed.
        x: str, column name where the UMAP X coordinates will be stored.
        y: str, column name where the UMAP Y coordinates will be stored.
        neighbors: str, column name where the nearest neighbor indices will be stored.
        model: str, name or path of the embedding model to use. The interpretation
            depends on the text_projector:
            - For 'sentence_transformers': A SentenceTransformer model name or path.
              See https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
            - For 'litellm': A LiteLLM-supported model identifier.
              See https://docs.litellm.ai/docs/embedding/supported_embedding
        batch_size: int, batch size for processing embeddings. Larger values use more
            memory but may be faster. Default is 32. When using 'litellm', note that
            different providers have different limits on input tokens per request, so
            batch size should be set accordingly to avoid exceeding API limits.
        text_projector: str, the embedding provider to use. Options are:
            - 'sentence_transformers' (default): Computes embeddings locally using the
              SentenceTransformers library. Requires the sentence-transformers package.
              This approach runs the model on your local machine and is suitable for
              privacy-sensitive data or when you want full control over the embedding
              process.
            - 'litellm': Computes embeddings remotely using API calls through the LiteLLM
              library. Requires the litellm package and appropriate API credentials.
              This approach supports a wide range of cloud-based embedding APIs
              (OpenAI, Cohere, Azure, etc.) and is useful when you want to leverage
              remote models without local computational overhead.
        umap_args: dict, additional keyword arguments to pass to the UMAP algorithm
            (e.g., n_neighbors, min_dist, metric).
        **kwargs: Additional configuration options for the embedding provider:
            - For 'sentence_transformers': Options passed to SentenceTransformer.encode().
              See https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
              Common options include 'trust_remote_code', 'normalize_embeddings', etc.
            - For 'litellm': Options passed to litellm.aembedding().
              See https://docs.litellm.ai/docs/embedding/supported_embedding#input-params-for-litellmembedding
              Common options include API keys, timeout settings, custom endpoints, etc.
              Special option 'sync' (bool, default False): Controls whether embeddings are
              processed asynchronously or synchronously. By default (False), batches are
              processed asynchronously, which is ideal for remote APIs like OpenAI, Vertex AI,
              etc., where the embedding happens on a remote instance and the task is IO-bound.
              However, when performing embeddings through a locally-running server (e.g., Ollama),
              setting sync=True can help avoid memory issues by processing batches sequentially.

    Returns:
        The input DataFrame with added columns for X, Y coordinates and nearest neighbors.
    """
    text_series = data_frame[text].astype(str).fillna("")
    text_projector_callback = _find_text_projector_callback(text_projector)

    proj = _projection_for_texts(
        list(text_series),
        model=model,
        batch_size=batch_size,
        text_projector_args=kwargs,
        text_projector=text_projector_callback,
        umap_args=umap_args,
    )
    data_frame[x] = proj.projection[:, 0]
    data_frame[y] = proj.projection[:, 1]
    if neighbors is not None:
        data_frame[neighbors] = [
            {"distances": b, "ids": a}  # ID is always the same as the row index.
            for a, b in zip(proj.knn_indices, proj.knn_distances)
        ]


def compute_vector_projection(
    data_frame: pd.DataFrame,
    vector: str,
    x: str = "projection_x",
    y: str = "projection_y",
    neighbors: str | None = "neighbors",
    umap_args: dict = {},
):
    """
    Generate 2D projections from pre-existing vector embeddings using UMAP.

    This function takes pre-computed vector embeddings and reduces their dimensionality
    to 2D coordinates using UMAP for visualization purposes.

    Args:
        data_frame: pandas DataFrame containing the vector data to process.
        vector: str, column name containing the pre-computed vector embeddings.
                Each entry should be a list or numpy array of numbers.
        x: str, column name where the UMAP X coordinates will be stored.
        y: str, column name where the UMAP Y coordinates will be stored.
        neighbors: str, column name where the nearest neighbor indices will be stored.
        umap_args: dict, additional keyword arguments to pass to the UMAP algorithm
            (e.g., n_neighbors, min_dist, metric).

    Returns:
        The input DataFrame with added columns for X, Y coordinates and nearest neighbors.
    """
    # Convert vector column to numpy array
    vector_series = data_frame[vector]

    # Convert each vector entry to numpy array and stack them
    vector_list = []
    for vector in vector_series:
        if isinstance(vector, list):
            vector_array = np.array(vector)
        elif isinstance(vector, np.ndarray):
            vector_array = vector
        else:
            # Try to convert to numpy array
            vector_array = np.array(vector)
        vector_list.append(vector_array)

    # Stack all vectors into a single numpy array
    hidden_vectors = np.stack(vector_list)

    # Run UMAP on the pre-existing vectors
    proj = _run_umap(hidden_vectors, umap_args)

    # Add projection results to dataframe
    data_frame[x] = proj.projection[:, 0]
    data_frame[y] = proj.projection[:, 1]
    if neighbors is not None:
        data_frame[neighbors] = [
            {"distances": b, "ids": a}  # ID is always the same as the row index.
            for a, b in zip(proj.knn_indices, proj.knn_distances)
        ]


def compute_image_projection(
    data_frame: pd.DataFrame,
    image: str,
    x: str = "projection_x",
    y: str = "projection_y",
    neighbors: str | None = "neighbors",
    model: str | None = None,
    trust_remote_code: bool = False,
    batch_size: int | None = None,
    umap_args: dict = {},
):
    """
    Compute image embeddings and generate 2D projections using UMAP.

    This function processes image data by creating embeddings using a model and
    then reducing the dimensionality to 2D coordinates using UMAP for
    visualization purposes.

    Args:
        data_frame: pandas DataFrame containing the image data to process.
        image: str, column name containing the images to embed.
        x: str, column name where the UMAP X coordinates will be stored.
        y: str, column name where the UMAP Y coordinates will be stored.
        neighbors: str, column name where the nearest neighbor indices will be stored.
        model: str, name or path of the model to use for embedding.
        trust_remote_code: bool, whether to trust and execute remote code when loading
            the model from HuggingFace Hub. Default is False.
        batch_size: int, batch size for processing images. Larger values use more
            memory but may be faster. Default is 16.
        umap_args: dict, additional keyword arguments to pass to the UMAP algorithm
            (e.g., n_neighbors, min_dist, metric).

    Returns:
        The input DataFrame with added columns for X, Y coordinates and nearest neighbors.
    """

    image_series = data_frame[image]
    proj = _projection_for_images(
        list(image_series),
        model=model,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        umap_args=umap_args,
    )
    data_frame[x] = proj.projection[:, 0]
    data_frame[y] = proj.projection[:, 1]
    if neighbors is not None:
        data_frame[neighbors] = [
            {"distances": b, "ids": a}  # ID is always the same as the row index.
            for a, b in zip(proj.knn_indices, proj.knn_distances)
        ]
