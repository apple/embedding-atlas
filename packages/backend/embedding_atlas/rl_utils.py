# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""
Utilities for working with RL replay buffer data.

This module implements replay buffer visualization inspired by:
"Vizarel: A System to Help Better Understand RL Agents" (Deshpande & Schneider, 2020)
https://arxiv.org/abs/2007.05577

The paper describes visualizing replay buffer experiences et = (st, at, rt, st+1)
by projecting them into 2D space using dimensionality reduction (t-SNE/UMAP).
"""

import logging
from typing import Any, Literal, Optional
import io

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ObsEncoderType = Literal["auto", "cnn", "pca", "raw"]


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def detect_rl_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Auto-detect common RL column patterns in a DataFrame.

    Supports both standard RL formats and Vizarel format:
    - Vizarel format (from paper): state, action, reward, next_state
    - Standard formats: obs/observation, action/act, reward/r

    Returns a dict with keys: 'observations', 'actions', 'rewards', 'other'
    """
    columns = list(df.columns)

    obs_patterns = ['obs', 'observation', 'state', 'next_state', 's_t', 'state_t']
    action_patterns = ['act', 'action', 'a_t', 'action_t']
    reward_patterns = ['reward', 'r', 'r_t', 'rew']

    detected: dict[str, list[str]] = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'other': []
    }

    for col in columns:
        col_lower = col.lower()
        categorized = False

        for pattern in obs_patterns:
            if pattern in col_lower:
                detected['observations'].append(col)
                categorized = True
                break

        if not categorized:
            for pattern in action_patterns:
                if pattern in col_lower:
                    detected['actions'].append(col)
                    categorized = True
                    break

        if not categorized:
            for pattern in reward_patterns:
                if pattern in col_lower:
                    detected['rewards'].append(col)
                    categorized = True
                    break

        if not categorized:
            detected['other'].append(col)

    return detected


# ---------------------------------------------------------------------------
# Image detection helpers
# ---------------------------------------------------------------------------

def is_image_shaped(value: Any) -> bool:
    """
    Return True if *value* is a numpy array with an image-like shape.

    Recognised layouts:
    - (H, W)        – grayscale
    - (H, W, C)     – HWC with C in {1, 3, 4} or stacked frames
    - (C, H, W)     – CHW (common in PyTorch-land)
    - (N, H, W)     – frame-stack where N is small (≤16)

    A minimum spatial size of 16 px is required on each axis to avoid
    mistaking short 1-D policy feature vectors for images.
    """
    if not isinstance(value, np.ndarray):
        return False
    s = value.shape
    if len(s) == 2:
        return s[0] >= 16 and s[1] >= 16
    if len(s) == 3:
        h, w, c = s[0], s[1], s[2]
        # HWC layout: spatial dims are first two
        if h >= 16 and w >= 16:
            return True
        # CHW layout: spatial dims are last two
        if w >= 16 and c >= 16:
            return True
    return False


def _obs_to_rgb_pil(obs: np.ndarray):
    """Convert an observation array to a (H, W, 3) uint8 PIL Image (RGB)."""
    from PIL import Image

    # Normalise dtype to uint8
    if obs.dtype != np.uint8:
        if obs.max() <= 1.0:
            obs = (obs * 255).astype(np.uint8)
        else:
            obs = np.clip(obs, 0, 255).astype(np.uint8)

    if len(obs.shape) == 2:
        # (H, W) – grayscale
        return Image.fromarray(obs, mode='L').convert('RGB')

    if len(obs.shape) == 3:
        h, w, c = obs.shape
        if h >= 16 and w >= 16:
            # HWC layout
            if c == 1:
                return Image.fromarray(obs[:, :, 0], mode='L').convert('RGB')
            if c == 3:
                return Image.fromarray(obs, mode='RGB')
            if c == 4:
                # Could be RGBA or a 4-frame stack; collapse to single channel
                return Image.fromarray(obs[:, :, 0], mode='L').convert('RGB')
            # Many stacked frames: average
            mean_frame = obs.mean(axis=2).astype(np.uint8)
            return Image.fromarray(mean_frame, mode='L').convert('RGB')
        else:
            # CHW layout
            if obs.shape[0] == 3:
                return Image.fromarray(obs.transpose(1, 2, 0), mode='RGB')
            return Image.fromarray(obs[0], mode='L').convert('RGB')

    raise ValueError(f"Unsupported observation shape for image conversion: {obs.shape}")


# ---------------------------------------------------------------------------
# Observation encoders
# ---------------------------------------------------------------------------

def encode_image_observations(
    observations: list[np.ndarray],
    encoder: ObsEncoderType = "auto",
    batch_size: int = 32,
    n_pca_components: int = 128,
) -> np.ndarray:
    """
    Encode a list of image-shaped observation arrays into compact feature vectors.

    Strategy selection:
    - ``"auto"``  – try CNN first; fall back to PCA if torch/torchvision missing.
    - ``"cnn"``   – lightweight pre-trained MobileNetV3-Small (576-dim features).
    - ``"pca"``   – PCA on flattened pixels (fast, no deep-learning deps).
    - ``"raw"``   – flatten pixels (original behaviour, **not recommended** for images).

    Returns
    -------
    np.ndarray of shape (N, feature_dim)
    """
    if encoder == "raw":
        logger.warning(
            "obs_encoder='raw' was selected. Flattening raw pixels for image "
            "observations. This produces very high-dimensional vectors and "
            "poor UMAP quality. Use 'cnn' or 'pca' for better results."
        )
        return np.stack([obs.flatten().astype(np.float32) for obs in observations])

    if encoder in ("auto", "cnn"):
        try:
            return _encode_with_cnn(observations, batch_size=batch_size)
        except Exception as exc:
            if encoder == "cnn":
                raise RuntimeError(
                    "CNN encoding failed. Ensure torch+torchvision are compatible."
                ) from exc
            logger.warning(
                "torch/torchvision not found; falling back to PCA encoding. "
                "Install them for better embedding quality: pip install torch torchvision"
            )

    # PCA fallback (also the explicit "pca" path)
    return _encode_with_pca(observations, n_components=n_pca_components)


def _encode_with_cnn(
    observations: list[np.ndarray],
    batch_size: int = 32,
    model_name: str | None = None,
) -> np.ndarray:
    """Dispatcher: tries torchvision first, then transformers pipeline."""
    import torch  # verify torch is available

    try:
        return _encode_with_torchvision(observations, batch_size, model_name)
    except Exception as tv_err:
        logger.warning(
            "torchvision encoder failed (%s); switching to transformers pipeline.",
            tv_err,
        )
        return _encode_with_transformers_pipeline(observations, batch_size, model_name)


def _encode_with_torchvision(
    observations: list[np.ndarray],
    batch_size: int = 32,
    model_name: str | None = None,
) -> np.ndarray:
    """
    Encode image observations using a pre-trained MobileNetV3-Small CNN.

    The classification head is removed so the output is a 576-dimensional
    feature vector per observation – small enough for fast UMAP but rich
    enough to capture meaningful visual structure.
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    import tqdm

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("CNN encoder using device: %s", device)

    if model_name is None:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)
    else:
        backbone = getattr(models, model_name)(weights="DEFAULT")

    # Strip the classifier; keep features + adaptive pool
    feature_extractor = torch.nn.Sequential(
        backbone.features,
        backbone.avgpool,
        torch.nn.Flatten(),
    )
    feature_extractor.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features: list[np.ndarray] = []
    n = len(observations)

    with torch.no_grad():
        for start in tqdm.tqdm(range(0, n, batch_size), desc="CNN encoding", unit="batch"):
            batch_obs = observations[start:start + batch_size]
            batch_imgs = [_obs_to_rgb_pil(obs) for obs in batch_obs]
            batch_tensor = torch.stack([transform(img) for img in batch_imgs]).to(device)
            features = feature_extractor(batch_tensor)
            all_features.append(features.cpu().numpy())

    result = np.concatenate(all_features, axis=0)
    logger.info(
        "CNN encoding complete: %d observations -> shape %s", n, result.shape
    )
    return result


def _encode_with_transformers_pipeline(
    observations: list[np.ndarray],
    batch_size: int = 32,
    model_name: str | None = None,
) -> np.ndarray:
    import torch
    import tqdm
    from transformers import pipeline

    if model_name is None:
        model_name = "google/vit-base-patch16-224"

    logger.info("Loading transformers image encoder: %s", model_name)
    pipe = pipeline("image-feature-extraction", model=model_name, device_map="auto")

    n = len(observations)
    all_features: list[np.ndarray] = []

    @torch.no_grad()
    def _run_batch(batch_imgs: list) -> None:
        rs = pipe(batch_imgs, return_tensors=True)
        for r in rs:
            r_t: torch.Tensor = r
            if r_t.dim() == 3:
                r_t = r_t.squeeze(0)
            if r_t.dim() == 2:
                r_t = r_t[0]
            all_features.append(r_t.cpu().float().numpy())

    current_batch: list = []
    for obs in tqdm.tqdm(observations, desc="transformers CNN", unit="obs"):
        current_batch.append(_obs_to_rgb_pil(obs))
        if len(current_batch) >= batch_size:
            _run_batch(current_batch)
            current_batch.clear()
    if current_batch:
        _run_batch(current_batch)

    result = np.stack(all_features, axis=0)
    logger.info("transformers encoding done: %d obs -> shape %s", n, result.shape)
    return result


def _encode_with_pca(
    observations: list[np.ndarray],
    n_components: int = 128,
) -> np.ndarray:
    """
    Reduce image observations to ``n_components`` dims using PCA.

    Faster than CNN but operates in pixel space, so the resulting embedding
    reflects visual similarity rather than semantic similarity.
    """
    from sklearn.decomposition import PCA

    flat = np.stack([obs.flatten().astype(np.float32) for obs in observations])
    if flat.max() > 1.0:
        flat = flat / 255.0

    n_components = min(n_components, flat.shape[0] - 1, flat.shape[1])
    logger.info(
        "PCA encoding: %d observations, %d -> %d dims",
        flat.shape[0], flat.shape[1], n_components,
    )
    pca = PCA(n_components=n_components, random_state=42)
    result = pca.fit_transform(flat)
    explained = pca.explained_variance_ratio_.sum()
    logger.info("PCA explains %.1f%% of variance", explained * 100)
    return result


# ---------------------------------------------------------------------------
# Vector extraction from arbitrary column values
# ---------------------------------------------------------------------------

def extract_vector_from_value(value: Any) -> np.ndarray:
    """Convert various formats to a flat numpy array."""
    if isinstance(value, np.ndarray):
        return value.flatten()
    elif isinstance(value, list):
        return np.array(value).flatten()
    elif isinstance(value, dict):
        if 'data' in value:
            return extract_vector_from_value(value['data'])
        elif 'array' in value:
            return extract_vector_from_value(value['array'])
        else:
            return np.array(list(value.values())).flatten()
    elif pd.isna(value):
        return np.array([])
    else:
        try:
            return np.array(value).flatten()
        except Exception:
            logger.warning("Could not convert value of type %s to array", type(value))
            return np.array([])


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def concatenate_columns_to_vectors(
    df: pd.DataFrame,
    columns: list[str],
    output_column: str = '__rl_vector__',
    obs_encoder: ObsEncoderType = "auto",
    encoder_batch_size: int = 32,
    pca_components: int = 128,
) -> pd.DataFrame:
    """
    Merge *columns* into a single vector column suitable for UMAP.

    Image-shaped columns (detected via :func:`is_image_shaped`) are encoded
    using a CNN or PCA before concatenation, avoiding the raw-pixel blowup.
    All other columns are flattened as before.
    """
    if not columns:
        raise ValueError("Must provide at least one column to concatenate")

    logger.info("Building projection vectors from columns: %s", columns)

    # Classify each column as 'image' or 'numeric'
    image_cols: list[str] = []
    numeric_cols: list[str] = []

    for col in columns:
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if sample is not None and is_image_shaped(sample):
            image_cols.append(col)
            logger.info(
                "Column '%s' contains image-shaped observations (shape %s) – "
                "will encode via %s encoder",
                col, sample.shape, obs_encoder,
            )
        else:
            numeric_cols.append(col)

    if image_cols:
        logger.info(
            "Image columns detected: %s. "
            "Using %s encoder instead of raw pixel flattening.",
            image_cols, obs_encoder,
        )

    # Pre-encode image columns: col_name -> np.ndarray of shape (N, feature_dim)
    encoded_images: dict[str, np.ndarray] = {}
    for col in image_cols:
        obs_list = list(df[col])
        encoded_images[col] = encode_image_observations(
            obs_list,
            encoder=obs_encoder,
            batch_size=encoder_batch_size,
            n_pca_components=pca_components,
        )

    # Build the final per-row vectors
    vectors: list[np.ndarray] = []
    for idx in range(len(df)):
        parts: list[np.ndarray] = []

        for col in columns:
            if col in encoded_images:
                parts.append(encoded_images[col][idx])
            else:
                parts.append(extract_vector_from_value(df[col].iloc[idx]))

        concatenated = np.concatenate(parts) if parts else np.array([])
        vectors.append(concatenated)

    lengths = {len(v) for v in vectors}
    if len(lengths) > 1:
        logger.warning(
            "Vectors have inconsistent lengths %s – check your data.", lengths
        )

    df[output_column] = vectors
    logger.info(
        "Projection vector column '%s' created with %d dims",
        output_column, len(vectors[0]) if vectors else 0,
    )
    return df


def prepare_rl_data_for_projection(
    df: pd.DataFrame,
    observation_columns: list[str] | None = None,
    action_columns: list[str] | None = None,
    concatenate: bool = True,
    vector_column: str = '__rl_vector__',
    preserve_image_columns: bool = True,
    obs_encoder: ObsEncoderType = "auto",
    encoder_batch_size: int = 32,
    pca_components: int = 128,
) -> tuple[pd.DataFrame, str]:
    """
    Prepare RL replay buffer data for UMAP projection.

    Parameters
    ----------
    df:
        DataFrame containing RL replay buffer data.
    observation_columns:
        Observation column names (auto-detected if ``None``).
    action_columns:
        Action column names (auto-detected if ``None``).
    concatenate:
        If ``True``, concatenate obs and action columns into one vector.
    vector_column:
        Output column name for the combined feature vector.
    preserve_image_columns:
        Keep image columns in the output DataFrame for tooltip display.
    obs_encoder:
        Encoding strategy for image-shaped observations.
        ``"auto"`` (default) tries CNN then falls back to PCA.
        ``"cnn"`` requires torch + torchvision.
        ``"pca"`` uses sklearn PCA – no GPU needed.
        ``"raw"`` flattens pixels (original behaviour, not recommended).
    encoder_batch_size:
        Batch size passed to the CNN encoder.
    pca_components:
        Number of PCA components when ``obs_encoder`` is ``"pca"`` or when
        CNN is unavailable and ``obs_encoder`` is ``"auto"``.

    Returns
    -------
    (processed_df, vector_column_name)
    """
    # Auto-detect columns if not provided
    if observation_columns is None and action_columns is None:
        logger.info("Auto-detecting RL columns...")
        detected = detect_rl_columns(df)
        observation_columns = detected['observations']
        action_columns = detected['actions']

        logger.info("Detected observations: %s", observation_columns)
        logger.info("Detected actions: %s", action_columns)
        logger.info("Detected rewards: %s", detected['rewards'])
        logger.info("Other columns: %s", detected['other'])

    # Identify image columns to preserve in output
    if preserve_image_columns:
        for col in df.columns:
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if sample is not None and (is_image_shaped(sample) or is_image_data(sample)):
                logger.info(
                    "Image column '%s' will be preserved for tooltip display.", col
                )

    columns_to_concat: list[str] = []
    if observation_columns:
        columns_to_concat.extend(observation_columns)
    if action_columns:
        columns_to_concat.extend(action_columns)

    if not columns_to_concat:
        raise ValueError(
            "No observation or action columns found. Please specify them explicitly "
            "via --observation-columns or --action-columns."
        )

    # Short-circuit: single non-image vector column
    if len(columns_to_concat) == 1 and not concatenate:
        single_col = columns_to_concat[0]
        first_value = df[single_col].iloc[0]
        if isinstance(first_value, (list, np.ndarray)) and not is_image_shaped(first_value):
            logger.info("Using existing vector column: %s", single_col)
            return df, single_col

    df = concatenate_columns_to_vectors(
        df,
        columns_to_concat,
        output_column=vector_column,
        obs_encoder=obs_encoder,
        encoder_batch_size=encoder_batch_size,
        pca_components=pca_components,
    )
    return df, vector_column


# ---------------------------------------------------------------------------
# Image data helpers (for Embedding Atlas format)
# ---------------------------------------------------------------------------

def is_image_data(value: Any) -> bool:
    """Check if a value is image data in Embedding Atlas format."""
    if value is None:
        return False
    if isinstance(value, dict) and "bytes" in value:
        return isinstance(value["bytes"], (bytes, np.ndarray))
    if isinstance(value, bytes):
        return True
    if isinstance(value, str) and value.startswith("data:image/"):
        return True
    return False


def capture_environment_image(env, mode: str = "rgb_array") -> Optional[bytes]:
    """Capture a rendered frame from a Gymnasium environment as PNG bytes."""
    try:
        if hasattr(env, 'render'):
            image_array = env.render()
            if image_array is not None:
                return convert_array_to_png_bytes(image_array)
        if hasattr(env, 'render'):
            try:
                image_array = env.render(mode=mode)
                if image_array is not None:
                    return convert_array_to_png_bytes(image_array)
            except Exception:
                pass
        if hasattr(env, 'unwrapped'):
            return capture_environment_image(env.unwrapped, mode)
    except Exception as e:
        logger.warning("Failed to capture environment image: %s", e)
    return None


def convert_array_to_png_bytes(image_array: np.ndarray) -> bytes:
    """Convert a numpy RGB array (H, W, 3) to PNG bytes."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image conversion. Install with: pip install Pillow"
        )

    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image = Image.fromarray(image_array, 'RGB')
    elif len(image_array.shape) == 2:
        image = Image.fromarray(image_array, 'L')
    else:
        raise ValueError(f"Unsupported image shape: {image_array.shape}")

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def image_bytes_to_embedding_atlas_format(image_bytes: bytes) -> dict:
    """Wrap image bytes in the dict format expected by Embedding Atlas."""
    return {"bytes": image_bytes}


def add_environment_images_to_replay_buffer(
    df: pd.DataFrame,
    env,
    image_column: str = "environment_image",
    max_images: Optional[int] = None,
    sample_strategy: str = "uniform",
) -> pd.DataFrame:
    """Add environment images to replay buffer by re-running episodes."""
    logger.info(
        "Adding environment images to replay buffer (strategy: %s)", sample_strategy
    )
    df[image_column] = None

    if max_images is None:
        indices_to_capture = list(range(len(df)))
    else:
        if sample_strategy == "uniform":
            step = max(1, len(df) // max_images)
            indices_to_capture = list(range(0, len(df), step))[:max_images]
        elif sample_strategy == "random":
            indices_to_capture = np.random.choice(
                len(df), size=min(max_images, len(df)), replace=False
            )
        elif sample_strategy == "episodes":
            if 'episode' in df.columns:
                episodes = df['episode'].unique()
                selected = episodes[:max_images] if len(episodes) >= max_images else episodes
                indices_to_capture = []
                for ep in selected:
                    ep_idx = df[df['episode'] == ep].index.tolist()
                    if ep_idx:
                        indices_to_capture.append(ep_idx[0])
            else:
                step = max(1, len(df) // max_images)
                indices_to_capture = list(range(0, len(df), step))[:max_images]
        else:
            raise ValueError(f"Unknown sample strategy: {sample_strategy}")

    captured = 0
    for idx in indices_to_capture:
        try:
            image_bytes = capture_environment_image(env)
            if image_bytes:
                df.at[idx, image_column] = image_bytes_to_embedding_atlas_format(image_bytes)
                captured += 1
        except Exception as e:
            logger.warning("Failed to capture image for index %d: %s", idx, e)

    logger.info("Captured %d images", captured)
    return df


# ---------------------------------------------------------------------------
# State unpacking & metadata helpers
# ---------------------------------------------------------------------------

def unpack_state_components(
    df: pd.DataFrame,
    state_col: str,
    component_names: list[str],
) -> pd.DataFrame:
    """
    Unpack a list-valued state column into individual named scalar columns.

    Example::

        unpack_state_components(df, "state",
            ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"])

    This turns ``state: [0.02, -0.1, 0.03, 0.5]`` into four columns.
    """
    first = df[state_col].iloc[0]
    vec = np.array(first).flatten()
    if len(component_names) != len(vec):
        raise ValueError(
            f"component_names has {len(component_names)} entries but state vectors "
            f"have {len(vec)} elements"
        )
    for i, name in enumerate(component_names):
        df[name] = df[state_col].apply(lambda s, idx=i: float(np.array(s).flatten()[idx]))
    logger.info("Unpacked '%s' into columns: %s", state_col, component_names)
    return df


def compute_td_error(
    df: pd.DataFrame,
    model,
    gamma: float = 0.99,
    state_col: str = "state",
    action_col: str = "action",
    reward_col: str = "reward",
    next_state_col: str = "next_state",
    done_col: str = "done",
) -> pd.DataFrame:
    """
    Compute TD error ``|r + gamma * max_Q(s') - Q(s, a)|`` for each transition.

    Requires a stable-baselines3 model with a ``q_net`` attribute (e.g. DQN).
    Adds a ``td_error`` column to *df* in-place.
    """
    import torch

    td_errors: list[float] = []
    with torch.no_grad():
        for _, row in df.iterrows():
            s = np.array(row[state_col], dtype=np.float32).reshape(1, -1)
            ns = np.array(row[next_state_col], dtype=np.float32).reshape(1, -1)
            s_t, _ = model.policy.obs_to_tensor(s)
            ns_t, _ = model.policy.obs_to_tensor(ns)
            q_s = model.q_net(s_t).cpu().numpy()[0]
            q_ns = model.q_net(ns_t).cpu().numpy()[0]
            q_sa = float(q_s[int(row[action_col])])
            max_q_ns = float(np.max(q_ns))
            done = float(row[done_col])
            target = float(row[reward_col]) + gamma * max_q_ns * (1.0 - done)
            td_errors.append(abs(target - q_sa))

    df["td_error"] = td_errors
    logger.info("Computed TD errors for %d transitions", len(td_errors))
    return df


def format_state_description(
    df: pd.DataFrame,
    component_cols: list[str],
    action_col: str = "action_name",
    q_col: str | None = "q_value",
    reward_col: str | None = "reward",
) -> pd.Series:
    """
    Build a concise human-readable string per row for the widget ``text`` column.

    Example output: ``"pole=3.2deg cart=0.1 push_right Q=12.3 R=1.0"``
    """
    def _fmt_row(row):
        parts = []
        for col in component_cols:
            val = row[col]
            parts.append(f"{col}={val:.2f}")
        if action_col and action_col in row.index:
            parts.append(str(row[action_col]))
        if q_col and q_col in row.index:
            parts.append(f"Q={row[q_col]:.2f}")
        if reward_col and reward_col in row.index:
            parts.append(f"R={row[reward_col]:.1f}")
        return " | ".join(parts)

    return df.apply(_fmt_row, axis=1)


def compute_episode_metadata(
    df: pd.DataFrame,
    done_col: str = "done",
    reward_col: str = "reward",
    timestep_col: str | None = "timestep",
) -> pd.DataFrame:
    """
    Compute episode-level metadata from replay buffer data.

    Adds columns: ``episode``, ``step_in_episode``, ``episode_length``,
    ``episode_return``, ``episode_phase``, ``cumulative_episode_reward``.
    """
    # Assign episode numbers
    ep, ep_list = 0, []
    for d in df[done_col]:
        ep_list.append(ep)
        if d:
            ep += 1
    df["episode"] = ep_list

    # Step within episode
    step_in_ep = []
    current_step = 0
    for d in df[done_col]:
        step_in_ep.append(current_step)
        current_step = 0 if d else current_step + 1
    df["step_in_episode"] = step_in_ep

    # Episode length
    ep_lengths = df.groupby("episode").size().to_dict()
    df["episode_length"] = df["episode"].map(ep_lengths)

    # Episode return (sum of rewards per episode)
    ep_returns = df.groupby("episode")[reward_col].sum().to_dict()
    df["episode_return"] = df["episode"].map(ep_returns)

    # Cumulative reward within episode
    cum_rewards = []
    running = 0.0
    for _, row in df.iterrows():
        running += float(row[reward_col])
        cum_rewards.append(running)
        if row[done_col]:
            running = 0.0
    df["cumulative_episode_reward"] = cum_rewards

    # Episode phase: start / middle / end
    def _phase(row):
        if row["episode_length"] <= 1:
            return "start"
        frac = row["step_in_episode"] / row["episode_length"]
        if frac < 0.2:
            return "start"
        elif frac > 0.8:
            return "end"
        return "middle"

    df["episode_phase"] = df.apply(_phase, axis=1)

    logger.info(
        "Computed episode metadata: %d episodes, lengths %d–%d",
        df["episode"].nunique(),
        df["episode_length"].min(),
        df["episode_length"].max(),
    )
    return df


def plot_episode_timeline(
    df: pd.DataFrame,
    episode_num: int,
    state_components: list[str],
    q_col: str = "q_value",
    action_col: str = "action_name",
):
    """
    Return a matplotlib figure showing state components and Q-values over time
    for a single episode.
    """
    import matplotlib.pyplot as plt

    ep_df = df[df["episode"] == episode_num].sort_values("step_in_episode")
    if len(ep_df) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.text(0.5, 0.5, f"Episode {episode_num} not found", ha="center", va="center")
        return fig

    n_plots = len(state_components) + (1 if q_col in ep_df.columns else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    steps = ep_df["step_in_episode"].values

    for i, comp in enumerate(state_components):
        axes[i].plot(steps, ep_df[comp].values, linewidth=1.5)
        axes[i].set_ylabel(comp, fontsize=9)
        axes[i].grid(True, alpha=0.3)

    if q_col in ep_df.columns:
        ax_q = axes[len(state_components)]
        ax_q.plot(steps, ep_df[q_col].values, color="tab:orange", linewidth=1.5)
        ax_q.set_ylabel(q_col, fontsize=9)
        ax_q.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step in Episode")
    fig.suptitle(f"Episode {episode_num} ({len(ep_df)} steps)", fontsize=12)
    fig.tight_layout()
    return fig


def capture_images_during_training(
    env,
    model,
    total_timesteps: int,
    capture_frequency: int = 100,
    image_column: str = "environment_image",
) -> list:
    """
    Return a callback object that captures rendered frames during training.

    Integrate this with your training loop or a stable-baselines3 callback.
    """
    captured_images: list = []

    logger.info(
        "Image capture callback ready (every %d steps)", capture_frequency
    )

    class ImageCaptureCallback:
        def __init__(self, env, frequency):
            self.env = env
            self.frequency = frequency
            self.step_count = 0

        def on_step(self) -> bool:
            self.step_count += 1
            if self.step_count % self.frequency == 0:
                image_bytes = capture_environment_image(self.env)
                if image_bytes:
                    captured_images.append((
                        self.step_count,
                        image_bytes_to_embedding_atlas_format(image_bytes),
                    ))
            return True

    return ImageCaptureCallback(env, capture_frequency)
