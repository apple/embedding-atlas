import asyncio
import random
from typing import Awaitable, Callable, TypeVar

from tqdm.auto import tqdm

from .utils import logger


class _BackoffState:
    def __init__(self, base_delay: float, max_delay: float):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.current_delay = 0.0
        self.consecutive_errors = 0

    def on_error(self):
        self.consecutive_errors += 1
        self.current_delay = min(
            self.max_delay, self.base_delay * (2 ** (self.consecutive_errors - 1))
        )

    def on_success(self):
        self.consecutive_errors = 0
        self.current_delay = 0.0


T = TypeVar("T")
R = TypeVar("R")


async def async_map(
    inputs: list[T],
    func: Callable[[T], Awaitable[R]],
    *,
    concurrency: int = 4,
    max_retry: int = 0,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 30.0,
    description: str = "Task",
    fallback: R | None = None,
) -> list[R]:
    """
    Map the inputs by an async function, return a future that resolves to the mapped array (in correct order).

    Args:
        inputs: List of items to process
        func: Async function to apply to each item
        concurrency: Maximum number of concurrent calls
        max_retry: Maximum number of retry attempts on failure (0 means no retries)
        retry_base_delay: Base delay in seconds for exponential backoff (default 1.0)
        retry_max_delay: Maximum delay in seconds for backoff cap (default 30.0)
        description: Description in the progress bar
        fallback: When an error happens, fill the given result. If None, raise the error.
                  When fallback is None and an error occurs, stops processing new tasks immediately.
    """
    count = len(inputs)
    results: list[R | None] = [None] * count
    semaphore = asyncio.Semaphore(concurrency)
    backoff = _BackoffState(retry_base_delay, retry_max_delay)
    # Event to signal that processing should stop (used when fallback is None and an error occurs)
    stop_event = asyncio.Event()
    # Store the first error encountered when fallback is None
    first_error: list[Exception | None] = [None]

    pbar = tqdm(total=count, desc=description)

    async def process(index: int, item: T) -> None:
        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(max_retry + 1):
                # Check if we should stop before each retry attempt
                if stop_event.is_set():
                    return

                try:
                    # All tasks respect the shared backoff
                    if backoff.current_delay > 0:
                        delay = random.uniform(0, backoff.current_delay)
                        logger.warning(
                            f"Backoff: waiting {delay:.1f}s before attempt {attempt + 1} for item {index}"
                        )
                        await asyncio.sleep(delay)
                    results[index] = await func(item)
                    backoff.on_success()
                    pbar.update(1)
                    return
                except Exception as e:
                    logger.error(e)
                    backoff.on_error()
                    last_error = e
                    if attempt < max_retry:
                        continue
            if last_error is not None:
                if fallback is None:
                    # Signal other tasks to stop and store the error
                    if first_error[0] is None:
                        first_error[0] = last_error
                    stop_event.set()
                else:
                    results[index] = fallback
                    pbar.update(1)

    await asyncio.gather(*(process(i, item) for i, item in enumerate(inputs)))

    pbar.close()

    # If we stopped due to an error, raise it
    if first_error[0] is not None:
        raise first_error[0]

    return results  # type: ignore[return-value]
