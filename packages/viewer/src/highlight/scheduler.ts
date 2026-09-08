// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Shared cross-instance batching queue around a {@link HighlightScorer}. Many
 * `use:highlight` instances calling for the same scorer + query within a
 * debounce window are coalesced into a single `scorer(uniqueTexts)` call;
 * identical texts are deduplicated and their results fanned back out.
 */

export interface HighlightSpan {
  /** Inclusive start character offset within the scored text. */
  start: number;
  /** Exclusive end character offset within the scored text. */
  end: number;
  /** Raw score; normalized to a heatmap bucket by the action. */
  value: number;
}

/**
 * Batched scoring function: one span array per input text, aligned by index.
 * `query` is the shared value the batch was collected under (see {@link schedule}).
 */
export type HighlightScorer<Q = unknown> = (texts: string[], query: Q) => Promise<HighlightSpan[][]>;

interface Waiter {
  resolve: (spans: HighlightSpan[]) => void;
  reject: (error: unknown) => void;
  signal?: AbortSignal;
  onAbort?: () => void;
}

interface Batch {
  byText: Map<string, Waiter[]>;
  timer: ReturnType<typeof setTimeout> | null;
  /** Query value passed to the scorer; shared by all texts under this key. */
  query: unknown;
}

const queues = new Map<HighlightScorer<any>, Map<string, Batch>>();

let objectKeyCounter = 0;
const objectKeys = new WeakMap<object, string>();

/**
 * Derive a stable string key from a `query` value, used both as the recompute
 * trigger and the batch discriminator. Equal primitives and identical object
 * references produce the same key (so they batch together).
 */
export function queryKeyOf(query: unknown): string {
  if (query == null) {
    return "";
  }
  if (typeof query === "object" || typeof query === "function") {
    let key = objectKeys.get(query as object);
    if (key == null) {
      key = `#${++objectKeyCounter}`;
      objectKeys.set(query as object, key);
    }
    return key;
  }
  return `${typeof query}:${String(query)}`;
}

/**
 * Queue `text` for scoring by `scorer` under `queryKey`. Resolves with the
 * spans for this text once the batch flushes. The `query` value is forwarded to
 * the scorer (all texts sharing a `queryKey` share the first-seen `query`).
 * Rejects with an AbortError if `signal` aborts before the flush commits.
 */
export function schedule(
  scorer: HighlightScorer<any>,
  queryKey: string,
  query: unknown,
  text: string,
  debounce: number,
  signal?: AbortSignal,
): Promise<HighlightSpan[]> {
  return new Promise<HighlightSpan[]>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }

    let byQuery = queues.get(scorer);
    if (byQuery == null) {
      byQuery = new Map();
      queues.set(scorer, byQuery);
    }
    let batch = byQuery.get(queryKey);
    if (batch == null) {
      batch = { byText: new Map(), timer: null, query };
      byQuery.set(queryKey, batch);
    }

    const waiter: Waiter = { resolve, reject, signal };
    let waiters = batch.byText.get(text);
    if (waiters == null) {
      waiters = [];
      batch.byText.set(text, waiters);
    }
    waiters.push(waiter);

    if (signal != null) {
      const onAbort = () => {
        const list = batch.byText.get(text);
        if (list != null) {
          const index = list.indexOf(waiter);
          if (index >= 0) {
            list.splice(index, 1);
          }
          if (list.length === 0) {
            batch.byText.delete(text);
          }
        }
        if (batch.byText.size === 0 && batch.timer != null) {
          clearTimeout(batch.timer);
          batch.timer = null;
          byQuery.delete(queryKey);
          if (byQuery.size === 0) {
            queues.delete(scorer);
          }
        }
        reject(new DOMException("Aborted", "AbortError"));
      };
      waiter.onAbort = onAbort;
      signal.addEventListener("abort", onAbort, { once: true });
    }

    if (batch.timer == null) {
      batch.timer = setTimeout(() => flush(scorer, queryKey), debounce);
    }
  });
}

function flush(scorer: HighlightScorer<any>, queryKey: string): void {
  const byQuery = queues.get(scorer);
  const batch = byQuery?.get(queryKey);
  if (byQuery == null || batch == null) {
    return;
  }

  // Detach this batch so new requests start a fresh one.
  byQuery.delete(queryKey);
  if (byQuery.size === 0) {
    queues.delete(scorer);
  }
  batch.timer = null;

  // We are committing: abort can no longer remove these waiters.
  for (const waiters of batch.byText.values()) {
    for (const waiter of waiters) {
      if (waiter.signal != null && waiter.onAbort != null) {
        waiter.signal.removeEventListener("abort", waiter.onAbort);
      }
    }
  }

  const unique = [...batch.byText.keys()];

  const rejectAll = (error: unknown): void => {
    for (const waiters of batch.byText.values()) {
      for (const waiter of waiters) {
        waiter.reject(error);
      }
    }
  };

  // `Promise.resolve().then` funnels a synchronous throw from a non-async scorer into the
  // same rejection path as an async failure, so waiters are never left unsettled (which would
  // hang `Promise.all` in the action). The length check enforces the one-array-per-text
  // contract instead of silently resolving dropped texts to empty highlights.
  Promise.resolve()
    .then(() => scorer(unique, batch.query))
    .then((results) => {
      if (results.length !== unique.length) {
        throw new Error(`HighlightScorer returned ${results.length} span arrays for ${unique.length} texts`);
      }
      unique.forEach((text, index) => {
        const spans = results[index] ?? [];
        batch.byText.get(text)?.forEach((waiter) => waiter.resolve(spans));
      });
    })
    .catch(rejectAll);
}
