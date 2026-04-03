# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""
Encrypted file-based caching module for embedding_atlas.

This module provides secure caching functionality with automatic encryption and
decryption of cached values. It supports caching arbitrary Python objects by
serializing them to JSON and encrypting the data using AES-GCM encryption.

Key features:
- Automatic encryption/decryption of cached data using AES-GCM
- Support for arbitrary Python objects (strings, dicts, lists, numpy arrays, etc.)
- Secure key derivation using HMAC and HKDF
- Atomic file operations to prevent corruption
- Configurable cache directory and serialization methods
- Two-level directory structure for efficient file organization

The cache uses a combination of HMAC-SHA256 for cache key generation and
HKDF-SHA256 for encryption key derivation, ensuring that cache keys and
encryption keys are cryptographically secure and derived from the input data.

Example:
    >>> from embedding_atlas.cache import file_cache_get, file_cache_set
    >>>
    >>> # Cache a value
    >>> file_cache_set("my_key", {"data": [1, 2, 3]})
    >>>
    >>> # Retrieve the cached value
    >>> cached_value = file_cache_get("my_key")
    >>> print(cached_value)  # {"data": [1, 2, 3]}
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import struct
from functools import lru_cache
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import IO, Any, Callable

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from platformdirs import user_cache_path

logger = logging.getLogger("embedding-atlas")


def file_cache_get(
    key: Any,
    *,
    scope: str | None = None,
    cache_root: str | Path | None = None,
    deserializer: Callable[[IO[bytes]], Any] | None = None,
) -> Any | None:
    """
    Retrieve a cached value from the encrypted file cache.

    This function attempts to load and decrypt a previously cached value using
    the provided key. The key is used to derive both the cache file location
    and the encryption key for decryption.

    Args:
        key: The cache key used to identify and decrypt the cached value.
             Can be any hashable type (str, bytes, dict, list, numpy array, etc.).
        cache_root: Optional custom cache directory path. If None, uses the
                   default user cache directory for embedding_atlas.
        deserializer: Optional custom function to deserialize the decrypted value
                     from a binary file descriptor. If None, uses JSON deserialization.

    Returns:
        The cached value if found and successfully decrypted, None otherwise.
        Returns None if the cache file doesn't exist, decryption fails, or
        deserialization fails.
    """
    cache_root = _resolve_cache_root(cache_root)
    if deserializer is None:
        deserializer = default_deserializer

    cache_key, encryption_key = _derive_cache_key_and_encryption_key(
        key, scope, cache_root
    )
    cache_path = cache_root / cache_key[:2] / cache_key

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as file:
            data = _decrypt_data(file.read(), key=encryption_key)

        return deserializer(BytesIO(data))
    except Exception:
        logger.debug("Cache read failed for key %s", cache_key, exc_info=True)
        return None


def file_cache_set(
    key: Any,
    value: Any,
    *,
    scope: str | None = None,
    cache_root: str | Path | None = None,
    serializer: Callable[[Any, IO[bytes]], None] | None = None,
):
    """
    Store a value in the encrypted file cache.

    This function serializes, encrypts, and stores a value in the file cache
    using the provided key. The key is used to derive both the cache file
    location and the encryption key. The operation is atomic - the file is
    written to a temporary location first, then renamed to prevent corruption.

    Args:
        key: The cache key used to identify and encrypt the cached value.
             Can be any hashable type (str, bytes, dict, list, numpy array, etc.).
        value: The value to cache. Must be serializable by the chosen serializer.
        cache_root: Optional custom cache directory path. If None, uses the
                   default user cache directory for embedding_atlas.
        serializer: Optional custom function to serialize the value to a binary
                   file descriptor before encryption. If None, uses JSON serialization.

    Raises:
        OSError: If there are issues creating cache directories or writing files.
        Exception: If serialization or encryption fails.
    """
    cache_root = _resolve_cache_root(cache_root)
    if serializer is None:
        serializer = default_serializer

    cache_key, encryption_key = _derive_cache_key_and_encryption_key(
        key, scope, cache_root
    )

    cache_path = cache_root / cache_key[:2] / cache_key

    # Generate a random temporary filename to avoid conflicts
    random_suffix = secrets.token_hex(8)
    cache_path_tmp = cache_root / cache_key[:2] / f"{cache_key}.tmp-{random_suffix}"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = BytesIO()
    serializer(value, buffer)
    encrypted_data = _encrypt_data(buffer.getvalue(), key=encryption_key)

    with open(cache_path_tmp, "wb") as file:
        file.write(encrypted_data)

    cache_path_tmp.rename(cache_path)


def file_cache_value(
    key: Any,
    value_func: Callable[[], Any],
    *,
    scope: str | None = None,
    cache_root: str | Path | None = None,
    serializer: Callable[[Any, IO[bytes]], None] | None = None,
    deserializer: Callable[[IO[bytes]], Any] | None = None,
    callback: Callable[[Path], None] | None = None,
):
    """
    Retrieve a cached value or compute and cache it if not present.

    This is a read-through cache helper: if a cached value exists for the given
    key, it is decrypted and returned. Otherwise, ``value_func`` is called to
    compute the value, which is then encrypted and stored before being returned.

    Args:
        key: The cache key used to locate and encrypt/decrypt the cached value.
             Can be any hashable type (str, bytes, dict, list, numpy array, etc.).
        value_func: A callable that takes no arguments and returns the value to
                    cache when a cache miss occurs.
        scope: Optional namespace to isolate cache entries. Different scopes
               produce different cache keys even for the same key value.
        cache_root: Optional custom cache directory path. If None, uses the
                   default user cache directory for embedding_atlas.
        serializer: Optional custom function to serialize the value to a binary
                   file descriptor before encryption. If None, uses JSON serialization.
        deserializer: Optional custom function to deserialize the decrypted value
                     from a binary file descriptor. If None, uses JSON deserialization.
        callback: Optional function called with the cache file path on a cache hit.

    Returns:
        The cached value on a hit, or the freshly computed value from
        ``value_func`` on a miss.
    """
    cache_root = _resolve_cache_root(cache_root)
    if serializer is None:
        serializer = default_serializer
    if deserializer is None:
        deserializer = default_deserializer

    cache_key, encryption_key = _derive_cache_key_and_encryption_key(
        key, scope, cache_root
    )

    cache_path = cache_root / cache_key[:2] / cache_key

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as file:
                data = _decrypt_data(file.read(), key=encryption_key)

            result = deserializer(BytesIO(data))

            if callback is not None:
                callback(cache_path)

            return result
        except Exception:
            # If we can't read the file, move on.
            logger.debug("Cache read failed for key %s", cache_key, exc_info=True)

    value = value_func()

    try:
        # Generate a random temporary filename to avoid conflicts
        random_suffix = secrets.token_hex(8)
        cache_path_tmp = cache_root / cache_key[:2] / f"{cache_key}.tmp-{random_suffix}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        buffer = BytesIO()
        serializer(value, buffer)
        encrypted_data = _encrypt_data(buffer.getvalue(), key=encryption_key)

        with open(cache_path_tmp, "wb") as file:
            file.write(encrypted_data)

        cache_path_tmp.rename(cache_path)
    except Exception:
        logger.debug("Cache write failed for key %s", cache_key, exc_info=True)

    return value


def _resolve_cache_root(cache_root: str | Path | None = None) -> Path:
    if cache_root is None:
        return (user_cache_path("embedding_atlas") / "cache").resolve()
    else:
        return Path(cache_root).resolve()


@lru_cache(maxsize=None)
def _get_constants(cache_root: Path) -> dict[str, bytes]:
    cache_root.mkdir(parents=True, exist_ok=True)
    constants_path = cache_root / "cache_constants.json"

    if constants_path.exists():
        with open(constants_path, "r") as f:
            data = json.load(f)
        return {
            "HMAC_KEY": base64.b64decode(data["HMAC_KEY"]),
            "HKDF_SALT": base64.b64decode(data["HKDF_SALT"]),
        }

    hmac_key = secrets.token_bytes(32)
    hkdf_salt = secrets.token_bytes(32)

    data = json.dumps(
        {
            "HMAC_KEY": base64.b64encode(hmac_key).decode("ascii"),
            "HKDF_SALT": base64.b64encode(hkdf_salt).decode("ascii"),
        },
        indent=2,
    ).encode("utf-8")

    try:
        # Use exclusive create ("xb") to avoid TOCTOU races.
        # open(..., "xb") fails with FileExistsError if the file already exists,
        # ensuring only one process wins when multiple start simultaneously.
        with open(constants_path, "xb") as f:
            f.write(data)
    except FileExistsError:
        # Another process created the file first — use their constants.
        with open(constants_path, "r") as f:
            existing = json.load(f)
        return {
            "HMAC_KEY": base64.b64decode(existing["HMAC_KEY"]),
            "HKDF_SALT": base64.b64decode(existing["HKDF_SALT"]),
        }

    return {"HMAC_KEY": hmac_key, "HKDF_SALT": hkdf_salt}


def _derive_cache_key_and_encryption_key(
    value: Any, scope: str | None, cache_root: Path
) -> tuple[str, bytes]:
    # First get sha256 of the value
    h = hashlib.sha256()
    _update_hash_with_value(h.update, scope, value)
    sha256 = h.digest()

    consts = _get_constants(cache_root)
    HMAC_KEY = consts["HMAC_KEY"]
    HKDF_SALT = consts["HKDF_SALT"]
    HKDF_INFO = b"cache-encryption-key"

    # HMAC for the cache key
    hasher = hmac.new(HMAC_KEY, digestmod=hashlib.sha256)
    hasher.update(sha256)
    cache_key = hasher.hexdigest()

    # HKDF for the encryption key
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=HKDF_SALT,
        info=HKDF_INFO,
    )
    encryption_key = hkdf.derive(sha256)

    return (cache_key, encryption_key)


def _encrypt_data(data: bytes, key: bytes) -> bytes:
    # Generate a random 96-bit (12 byte) nonce for GCM
    nonce = secrets.token_bytes(12)

    # Create cipher
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
    encryptor = cipher.encryptor()

    # Encrypt the data
    ciphertext = encryptor.update(data) + encryptor.finalize()

    # Return nonce + tag + ciphertext
    return nonce + encryptor.tag + ciphertext


def _decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    # Extract nonce (12 bytes), tag (16 bytes), and ciphertext
    nonce = encrypted_data[:12]
    tag = encrypted_data[12:28]
    ciphertext = encrypted_data[28:]

    # Create cipher
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag))
    decryptor = cipher.decryptor()

    # Decrypt the data
    return decryptor.update(ciphertext) + decryptor.finalize()


def _update_hash_with_value(update_func: Callable[[bytes], None], *value: Any):
    def preamble(kind: bytes, length: int):
        update_func(kind + b":" + struct.pack("<Q", length))

    def emit(kind: bytes, data: bytes):
        preamble(kind, len(data))
        update_func(data)

    def emit_value(v):
        if v is None:
            preamble(b"null", 0)
        elif isinstance(v, bytes):
            emit(b"bytes", v)
        elif isinstance(v, str):
            emit(b"str", v.encode("utf-8"))
        elif isinstance(v, np.ndarray):
            prefix_bytes = (
                v.dtype.str.encode("ascii")
                + b"\x00"
                + struct.pack(f"<I{len(v.shape)}Q", len(v.shape), *v.shape)
            )
            data_bytes = v.tobytes()
            preamble(b"np.ndarray", len(prefix_bytes) + len(data_bytes))
            update_func(prefix_bytes)
            update_func(data_bytes)
        elif isinstance(v, list):
            preamble(b"list", len(v))
            for item in v:
                emit_value(item)
        elif isinstance(v, dict):
            preamble(b"dict", len(v))
            for k in sorted(v.keys(), key=str):
                emit_value(k)
                emit_value(v[k])
        else:
            emit(b"json", json.dumps(v, sort_keys=True).encode("utf-8"))

    for item in value:
        emit_value(item)


def sha256_hexdigest(value: Any, scope: str | None = None):
    h = hashlib.sha256()
    _update_hash_with_value(h.update, scope, value)
    return h.hexdigest()


def default_serializer(value: Any, fd: IO[bytes]) -> None:
    text_fd = TextIOWrapper(fd, encoding="utf-8")
    json.dump(value, text_fd)
    text_fd.detach()


def default_deserializer(fd: IO[bytes]) -> Any:
    text_fd = TextIOWrapper(fd, encoding="utf-8")
    result = json.load(text_fd)
    text_fd.detach()
    return result
