# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

from collections.abc import Callable
from io import BytesIO
from typing import Any

import numpy as np

from .utils import logger


def create_embedder(
    name: str, *, modality: str, model: str | None, embedder_args: dict
) -> Callable:
    """Create a built-in embedder function by name."""
    factories = {
        "sentence-transformers": _create_sentence_transformers_embedder,
        "transformers": _create_transformers_embedder,
        "litellm": _create_litellm_embedder,
    }
    if name not in factories:
        raise ValueError(
            f"Unknown embedder: {name}. Must be one of: {list(factories.keys())}"
        )
    return factories[name](modality=modality, model=model, embedder_args=embedder_args)


def _create_sentence_transformers_embedder(
    *, modality: str, model: str | None, embedder_args: dict
) -> Callable:
    """Return an async embedder backed by SentenceTransformers (text only)."""
    if modality != "text":
        raise NotImplementedError(
            "The sentence-transformers embedder only supports text embedding"
        )

    from sentence_transformers import SentenceTransformer

    model_name = model or "all-MiniLM-L6-v2"
    default_args = {"trust_remote_code": False}
    merged = {**default_args, **embedder_args}
    logger.info("Loading model %s...", model_name)
    st_model = SentenceTransformer(model_name, **merged)

    async def _embed(
        batch: list[str], *, model: str | None, embedder_args: dict
    ) -> np.ndarray:
        return st_model.encode(batch, show_progress_bar=False, batch_size=len(batch))

    return _embed


def _create_transformers_embedder(
    *, modality: str, model: str | None, embedder_args: dict
) -> Callable:
    """Return an async embedder backed by HuggingFace transformers pipelines."""
    dispatch = {
        "text": _create_transformers_text_embedder,
        "image": _create_transformers_image_embedder,
        "audio": _create_transformers_audio_embedder,
    }
    if modality not in dispatch:
        raise NotImplementedError(
            f"The transformers embedder does not support {modality} embeddings"
        )
    return dispatch[modality](model=model, embedder_args=embedder_args)


def _create_transformers_text_embedder(
    *, model: str | None, embedder_args: dict
) -> Callable:
    """Return an async embedder backed by a HuggingFace feature-extraction pipeline."""
    from transformers import pipeline

    model_name = model or "sentence-transformers/all-MiniLM-L6-v2"
    logger.info("Loading transformers pipeline for model %s...", model_name)
    pipe = pipeline("feature-extraction", model=model_name, **embedder_args)

    async def _embed(
        batch: list[Any], *, model: str | None, embedder_args: dict
    ) -> np.ndarray:
        outputs = pipe(batch)
        embeddings = []
        for output in outputs:
            arr = np.array(output)
            if arr.ndim > 1:
                arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
            embeddings.append(arr)
        return np.stack(embeddings).astype(np.float32)

    return _embed


def _create_transformers_image_embedder(
    *, model: str | None, embedder_args: dict
) -> Callable:
    """Return an async embedder backed by a HuggingFace image-feature-extraction pipeline."""
    from transformers import pipeline

    model_name = model or "google/vit-base-patch16-224"
    logger.info("Loading transformers pipeline for model %s...", model_name)
    pipe = pipeline("image-feature-extraction", model=model_name, **embedder_args)

    async def _embed(
        batch: list[Any], *, model: str | None, embedder_args: dict
    ) -> np.ndarray:
        from PIL import Image

        images = [Image.open(BytesIO(item["bytes"])).convert("RGB") for item in batch]
        outputs = pipe(images)  # type: ignore
        embeddings = []
        for output in outputs:
            arr = np.array(output)
            if arr.ndim > 1:
                arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
            embeddings.append(arr)
        return np.stack(embeddings).astype(np.float32)

    return _embed


def _create_transformers_audio_embedder(
    *, model: str | None, embedder_args: dict
) -> Callable:
    """Return an async embedder backed by CLAP for audio data."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "Audio embedding requires the `soundfile` package. "
            "Please run `pip install soundfile`, then try again."
        ) from None

    import torch
    from transformers import ClapModel, ClapProcessor

    model_name = model or "laion/clap-htsat-fused"
    logger.info("Loading CLAP model %s...", model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_model = ClapModel.from_pretrained(model_name, **embedder_args).to(device)  # type: ignore
    clap_processor = ClapProcessor.from_pretrained(model_name)

    target_sr = clap_processor.feature_extractor.sampling_rate  # type: ignore

    async def _embed(
        batch: list[Any], *, model: str | None, embedder_args: dict
    ) -> np.ndarray:
        from scipy.signal import resample

        waveforms = []
        for item in batch:
            audio_bytes = item["bytes"]
            data, sr = sf.read(BytesIO(audio_bytes))
            # Convert stereo to mono
            if data.ndim > 1:
                data = data.mean(axis=1)
            # Resample to expected rate if needed
            if sr != target_sr:
                num_samples = int(len(data) * target_sr / sr)
                data = resample(data, num_samples)
            waveforms.append(data)

        inputs = clap_processor(
            audio=waveforms,
            sampling_rate=target_sr,  # type: ignore
            return_tensors="pt",  # type: ignore
            padding=True,  # type: ignore
        ).to(device)

        with torch.no_grad():
            audio_embeds = clap_model.get_audio_features(**inputs)

        if hasattr(audio_embeds, "pooler_output"):
            audio_embeds = audio_embeds.pooler_output  # type: ignore
        return audio_embeds.cpu().float().numpy()  # type: ignore

    return _embed


def _create_litellm_embedder(
    *, modality: str, model: str | None, embedder_args: dict
) -> Callable:
    """Return an async embedder backed by LiteLLM."""

    if model is None:
        raise ValueError("model must be specified with the litellm embedder")

    async def _embed(
        batch: list[Any], *, model: str | None, embedder_args: dict
    ) -> np.ndarray:
        from litellm import aembedding

        if model is None:
            raise ValueError("model must be specified with the litellm embedder")

        if modality == "image":
            import base64

            embeddings = []
            for item in batch:
                b64 = base64.b64encode(item["bytes"]).decode("ascii")
                response = await aembedding(
                    input=[f"data:image/png;base64,{b64}"],
                    model=model,
                    **embedder_args,
                )
                embeddings.append(response.data[0]["embedding"])
            return np.array(embeddings)
        else:
            response = await aembedding(
                input=batch,
                model=model,
                **embedder_args,
            )
            return np.array([item["embedding"] for item in response.data])

    return _embed
