import asyncio

import pytest
from embedding_atlas.async_map import async_map


class TestAsyncMap:
    """Tests for the async_map function."""

    @pytest.mark.asyncio
    async def test_basic_mapping(self):
        """Test basic async mapping with a simple function."""
        inputs = [1, 2, 3, 4, 5]

        async def double(x: int) -> int:
            return x * 2

        results = await async_map(inputs, double, description="Doubling")
        assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test async_map with empty input list."""
        inputs: list[int] = []

        async def double(x: int) -> int:
            return x * 2

        results = await async_map(inputs, double, description="Empty")
        assert results == []

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        """Test that results are returned in the correct order despite async execution."""
        inputs = [5, 1, 3, 2, 4]

        async def delayed_identity(x: int) -> int:
            # Shorter delays complete first, but order should be preserved
            await asyncio.sleep(x * 0.001)
            return x

        results = await async_map(
            inputs, delayed_identity, concurrency=5, description="Order test"
        )
        assert results == [5, 1, 3, 2, 4]

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency limit is respected."""
        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrency(x: int) -> int:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return x

        inputs = list(range(10))
        await async_map(
            inputs, track_concurrency, concurrency=3, description="Concurrency test"
        )

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that retries work when function fails."""
        attempt_counts: dict[int, int] = {}

        async def fail_twice(x: int) -> int:
            attempt_counts[x] = attempt_counts.get(x, 0) + 1
            if attempt_counts[x] <= 2:
                raise ValueError(f"Attempt {attempt_counts[x]} for {x}")
            return x * 2

        inputs = [1, 2, 3]
        results = await async_map(
            inputs,
            fail_twice,
            max_retry=3,
            retry_base_delay=0,
            description="Retry test",
        )

        assert results == [2, 4, 6]
        # Each item should have been attempted 3 times (2 failures + 1 success)
        for x in inputs:
            assert attempt_counts[x] == 3

    @pytest.mark.asyncio
    async def test_no_retry_raises_immediately(self):
        """Test that without retries, errors are raised immediately."""

        async def always_fail(x: int) -> int:
            raise ValueError(f"Failed for {x}")

        inputs = [1, 2, 3]
        with pytest.raises(ValueError, match="Failed for"):
            await async_map(
                inputs, always_fail, max_retry=0, description="No retry test"
            )

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """Test that fallback value is used when function fails and fallback is provided."""

        async def fail_on_even(x: int) -> int:
            if x % 2 == 0:
                raise ValueError(f"Even number: {x}")
            return x * 2

        inputs = [1, 2, 3, 4, 5]
        results = await async_map(
            inputs,
            fail_on_even,
            fallback=-1,
            retry_base_delay=0,
            description="Fallback test",
        )

        assert results == [2, -1, 6, -1, 10]

    @pytest.mark.asyncio
    async def test_fallback_with_retry(self):
        """Test that fallback is used after all retries are exhausted."""
        attempt_counts: dict[int, int] = {}

        async def always_fail(x: int) -> int:
            attempt_counts[x] = attempt_counts.get(x, 0) + 1
            raise ValueError(f"Always fails for {x}")

        inputs = [1, 2]
        results = await async_map(
            inputs,
            always_fail,
            max_retry=2,
            retry_base_delay=0,
            fallback=0,
            description="Fallback retry test",
        )

        assert results == [0, 0]
        # Each item should have been attempted 3 times (initial + 2 retries)
        for x in inputs:
            assert attempt_counts[x] == 3

    @pytest.mark.asyncio
    async def test_string_inputs(self):
        """Test async_map with string inputs."""
        inputs = ["hello", "world", "test"]

        async def uppercase(s: str) -> str:
            return s.upper()

        results = await async_map(inputs, uppercase, description="Uppercase")
        assert results == ["HELLO", "WORLD", "TEST"]

    @pytest.mark.asyncio
    async def test_different_return_type(self):
        """Test async_map where return type differs from input type."""
        inputs = [1, 2, 3]

        async def to_string(x: int) -> str:
            return f"num_{x}"

        results = await async_map(inputs, to_string, description="To string")
        assert results == ["num_1", "num_2", "num_3"]

    @pytest.mark.asyncio
    async def test_stops_early_on_error_without_fallback(self):
        """Test that processing stops early when an error occurs and fallback is None."""
        processed_items: list[int] = []
        started_items: list[int] = []

        async def slow_fail_on_first(x: int) -> int:
            started_items.append(x)
            if x == 1:
                # First item fails immediately
                raise ValueError(f"Failed for {x}")
            # Other items take longer to process
            await asyncio.sleep(0.02)
            processed_items.append(x)
            return x * 2

        # Use many inputs with low concurrency to ensure some haven't started yet
        inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        with pytest.raises(ValueError, match="Failed for 1"):
            await async_map(
                inputs,
                slow_fail_on_first,
                concurrency=2,
                description="Early stop test",
            )

        # Not all items should have been processed since we stopped early
        # With concurrency=2, items 1 and 2 start first. Item 1 fails immediately,
        # which should prevent most other items from being fully processed.
        assert (
            len(processed_items) < len(inputs) - 1
        )  # At least some items were skipped

    @pytest.mark.asyncio
    async def test_does_not_stop_early_with_fallback(self):
        """Test that processing continues when fallback is provided even on errors."""
        processed_items: list[int] = []

        async def fail_on_even(x: int) -> int:
            await asyncio.sleep(0.001)
            if x % 2 == 0:
                raise ValueError(f"Even number: {x}")
            processed_items.append(x)
            return x * 2

        inputs = [1, 2, 3, 4, 5]
        results = await async_map(
            inputs,
            fail_on_even,
            fallback=-1,
            concurrency=2,
            description="Continue with fallback test",
        )

        # All odd items should have been processed
        assert sorted(processed_items) == [1, 3, 5]
        # Results should include fallback for even numbers
        assert results == [2, -1, 6, -1, 10]

    @pytest.mark.asyncio
    async def test_backoff_slows_down_after_errors(self):
        """Test that errors cause subsequent calls to be delayed via shared backoff."""
        call_times: list[float] = []
        attempt_counts: dict[int, int] = {}

        async def fail_then_succeed(x: int) -> int:
            attempt_counts[x] = attempt_counts.get(x, 0) + 1
            call_times.append(asyncio.get_event_loop().time())
            if x == 0 and attempt_counts[x] <= 2:
                raise ValueError("transient error")
            return x

        # Item 0 fails twice then succeeds, items 1-2 should be delayed by the shared backoff.
        # Use concurrency=1 so calls are sequential and timing is deterministic.
        start = asyncio.get_event_loop().time()
        results = await async_map(
            [0, 0, 0],
            fail_then_succeed,
            concurrency=1,
            max_retry=3,
            retry_base_delay=0.05,
            retry_max_delay=1.0,
            description="Backoff test",
        )

        elapsed = asyncio.get_event_loop().time() - start
        assert results == [0, 0, 0]
        # Two failures with base_delay=0.05 means delays of up to 0.05s and 0.1s.
        assert elapsed > 0.02, "Expected some backoff delay"

    @pytest.mark.asyncio
    async def test_backoff_resets_on_success(self):
        """Test that a successful call resets the shared backoff so later calls aren't delayed."""
        attempt_counts: dict[int, int] = {}

        async def fail_first_attempt(x: int) -> int:
            attempt_counts[x] = attempt_counts.get(x, 0) + 1
            if x == 0 and attempt_counts[x] == 1:
                raise ValueError("one-time error")
            return x

        # Item 0 fails once then succeeds (resetting backoff), items 1-4 should not be delayed.
        # Use concurrency=1 for sequential execution.
        start = asyncio.get_event_loop().time()
        results = await async_map(
            [0, 1, 2, 3, 4],
            fail_first_attempt,
            concurrency=1,
            max_retry=1,
            retry_base_delay=0.1,
            retry_max_delay=1.0,
            description="Backoff reset test",
        )

        elapsed = asyncio.get_event_loop().time() - start
        assert results == [0, 1, 2, 3, 4]
        # If backoff didn't reset, items 1-4 would each wait ~0.1s (total ~0.4s+).
        # With reset, only item 0's retry has a delay.
        assert elapsed < 0.5, "Backoff should have reset after success"

    @pytest.mark.asyncio
    async def test_backoff_affects_concurrent_tasks(self):
        """Test that backoff from one task's failure delays other concurrent tasks."""
        call_times: dict[int, list[float]] = {}

        async def fail_item_0(x: int) -> int:
            t = asyncio.get_event_loop().time()
            call_times.setdefault(x, []).append(t)
            if x == 0:
                raise ValueError("always fails")
            # Item 1 takes a moment so item 0 fails first
            await asyncio.sleep(0.01)
            return x

        # Item 0 always fails (with fallback so processing continues).
        # Item 1 should be delayed on its second invocation due to shared backoff from item 0's failure.
        results = await async_map(
            [0, 0, 1, 1],
            fail_item_0,
            concurrency=4,
            max_retry=0,
            retry_base_delay=0.05,
            retry_max_delay=1.0,
            fallback=-1,
            description="Shared backoff test",
        )

        assert results[2] == 1 or results[2] == -1  # item at index 2 has input 1
        # The key assertion: later calls should have been delayed.
        # With 4 concurrent tasks and backoff, total time should exceed just the sleep time.
        all_times = [t for times in call_times.values() for t in times]
        spread = max(all_times) - min(all_times)
        assert spread > 0.01, "Expected backoff to spread out call times"
