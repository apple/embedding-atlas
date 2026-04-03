import numpy as np
import pytest
from embedding_atlas.cache import (
    file_cache_get,
    file_cache_set,
    file_cache_value,
    sha256_hexdigest,
)

# ---------------------------------------------------------------------------
# sha256_hexdigest
# ---------------------------------------------------------------------------


def test_sha256_hexdigest_deterministic():
    assert sha256_hexdigest("hello") == sha256_hexdigest("hello")


def test_sha256_hexdigest_different_values():
    assert sha256_hexdigest("a") != sha256_hexdigest("b")


def test_sha256_hexdigest_with_scope():
    h1 = sha256_hexdigest("data", scope="scope1")
    h2 = sha256_hexdigest("data", scope="scope2")
    h3 = sha256_hexdigest("data", scope=None)
    assert h1 != h2
    assert h1 != h3


def test_sha256_hexdigest_bytes():
    h = sha256_hexdigest(b"raw bytes")
    assert isinstance(h, str) and len(h) == 64


def test_sha256_hexdigest_numpy_array():
    arr = np.array([1.0, 2.0, 3.0])
    h1 = sha256_hexdigest(arr)
    h2 = sha256_hexdigest(np.array([1.0, 2.0, 3.0]))
    h3 = sha256_hexdigest(np.array([1.0, 2.0, 4.0]))
    assert h1 == h2
    assert h1 != h3


def test_sha256_hexdigest_list():
    assert sha256_hexdigest([1, 2, 3]) == sha256_hexdigest([1, 2, 3])
    assert sha256_hexdigest([1, 2, 3]) != sha256_hexdigest([1, 2, 4])


def test_sha256_hexdigest_dict():
    # Dict hashing should be order-independent (keys are sorted)
    assert sha256_hexdigest({"a": 1, "b": 2}) == sha256_hexdigest({"b": 2, "a": 1})


def test_sha256_hexdigest_none():
    h = sha256_hexdigest(None)
    assert isinstance(h, str) and len(h) == 64
    assert h != sha256_hexdigest("")


def test_sha256_hexdigest_nested():
    val = {"key": [1, "two", None, {"inner": True}]}
    assert sha256_hexdigest(val) == sha256_hexdigest(val)


# ---------------------------------------------------------------------------
# file_cache_get / file_cache_set / file_cache_value  (integration tests)
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache_dir(tmp_path):
    """Provide a fresh temporary cache directory and clear the constants LRU cache."""
    import embedding_atlas.cache as cache_module

    cache_module._get_constants.cache_clear()
    yield tmp_path / "cache"
    cache_module._get_constants.cache_clear()


def test_file_cache_miss(cache_dir):
    assert file_cache_get("nonexistent", cache_root=cache_dir) is None


def test_file_cache_set_and_get(cache_dir):
    file_cache_set("key1", {"value": 42}, cache_root=cache_dir)
    result = file_cache_get("key1", cache_root=cache_dir)
    assert result == {"value": 42}


def test_file_cache_set_overwrite(cache_dir):
    file_cache_set("key", "old", cache_root=cache_dir)
    file_cache_set("key", "new", cache_root=cache_dir)
    assert file_cache_get("key", cache_root=cache_dir) == "new"


def test_file_cache_different_keys(cache_dir):
    file_cache_set("a", 1, cache_root=cache_dir)
    file_cache_set("b", 2, cache_root=cache_dir)
    assert file_cache_get("a", cache_root=cache_dir) == 1
    assert file_cache_get("b", cache_root=cache_dir) == 2


def test_file_cache_with_scope(cache_dir):
    file_cache_set("key", "val1", scope="s1", cache_root=cache_dir)
    file_cache_set("key", "val2", scope="s2", cache_root=cache_dir)
    assert file_cache_get("key", scope="s1", cache_root=cache_dir) == "val1"
    assert file_cache_get("key", scope="s2", cache_root=cache_dir) == "val2"
    # Wrong scope returns None
    assert file_cache_get("key", scope="s3", cache_root=cache_dir) is None


def test_file_cache_value_miss(cache_dir):
    calls = []

    def compute():
        calls.append(1)
        return "computed"

    result = file_cache_value("k", compute, cache_root=cache_dir)
    assert result == "computed"
    assert len(calls) == 1


def test_file_cache_value_hit(cache_dir):
    file_cache_set("k", "cached", cache_root=cache_dir)
    calls = []

    def compute():
        calls.append(1)
        return "computed"

    result = file_cache_value("k", compute, cache_root=cache_dir)
    assert result == "cached"
    assert len(calls) == 0


def test_file_cache_value_callback(cache_dir):
    file_cache_set("k", "cached", cache_root=cache_dir)
    paths = []
    result = file_cache_value(
        "k", lambda: "x", cache_root=cache_dir, callback=lambda p: paths.append(p)
    )
    assert result == "cached"
    assert len(paths) == 1


def test_file_cache_value_no_callback_on_miss(cache_dir):
    paths = []
    file_cache_value(
        "k", lambda: "x", cache_root=cache_dir, callback=lambda p: paths.append(p)
    )
    assert len(paths) == 0


def test_file_cache_custom_serializer(cache_dir):
    def ser(v, fd):
        fd.write(v.encode("ascii"))

    def deser(fd):
        return fd.read().decode("ascii")

    file_cache_set("k", "hello", cache_root=cache_dir, serializer=ser)
    result = file_cache_get("k", cache_root=cache_dir, deserializer=deser)
    assert result == "hello"


def test_file_cache_complex_key(cache_dir):
    key = {"model": "bert", "params": [1, 2], "array": np.array([1.0, 2.0])}
    file_cache_set(key, "result", cache_root=cache_dir)
    assert file_cache_get(key, cache_root=cache_dir) == "result"


def test_file_cache_value_populates_cache(cache_dir):
    file_cache_value("k", lambda: [1, 2, 3], cache_root=cache_dir)
    # Should be readable via file_cache_get now
    assert file_cache_get("k", cache_root=cache_dir) == [1, 2, 3]


def test_cache_files_are_encrypted(cache_dir):
    file_cache_set("key", "secret_value", cache_root=cache_dir)
    # Find the cache file and verify contents are not plaintext
    cache_files = list(cache_dir.rglob("*"))
    data_files = [
        f for f in cache_files if f.is_file() and f.name != "cache_constants.json"
    ]
    assert len(data_files) == 1
    raw = data_files[0].read_bytes()
    assert b"secret_value" not in raw
