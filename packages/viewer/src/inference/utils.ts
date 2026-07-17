// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { delay } from "@embedding-atlas/utils";

/**
 * Split `inputs` into successive `size`-long chunks. The last chunk may be
 * shorter; an empty input array yields an empty array of chunks.
 */
export function chunkInputs<T>(inputs: T[], size: number): T[][] {
  if (size < 1) {
    throw new Error(`chunkInputs: size must be >= 1, got ${size}`);
  }
  const chunks: T[][] = [];
  for (let start = 0; start < inputs.length; start += size) {
    chunks.push(inputs.slice(start, start + size));
  }
  return chunks;
}

/**
 * Run `embedChunk` over `chunks` with a worker pool. The pool size is
 * initialized from `getConcurrency()`, and each worker re-reads the limit
 * before pulling its next chunk — if its index is now above the cap (e.g. AIMD
 * shrunk after a 429), the worker exits gracefully after finishing its current
 * chunk. Worker 0 always continues, since `getConcurrency()` is floored at 1.
 *
 * Results are placed back by index, so the output order matches `chunks`
 * regardless of which chunk finishes first. A rejection from `embedChunk`
 * propagates out via `Promise.all`.
 */
export async function dispatchPool<T, R>(
  chunks: T[][],
  embedChunk: (chunk: T[]) => Promise<R>,
  getConcurrency: () => number,
): Promise<R[]> {
  const parts: R[] = new Array(chunks.length);
  let next = 0;
  const worker = async (idx: number) => {
    while (true) {
      if (idx >= Math.max(1, getConcurrency())) {
        return;
      }
      const i = next++;
      if (i >= chunks.length) {
        return;
      }
      parts[i] = await embedChunk(chunks[i]);
    }
  };
  const initial = Math.min(Math.max(1, getConcurrency()), chunks.length);
  await Promise.all(Array.from({ length: initial }, (_, i) => worker(i)));
  return parts;
}

/**
 * Error thrown by HTTP-backed providers when the server returns a non-2xx
 * status. Carries the status code and an optional `Retry-After`-derived delay
 * so retry policies can react without re-parsing the response.
 */
export class HttpError extends Error {
  status: number;
  retryAfterMs?: number;

  constructor(message: string, status: number, retryAfterMs?: number) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.retryAfterMs = retryAfterMs;
  }
}

/**
 * Upper bound on a parsed `Retry-After` delay. A malformed or malicious header
 * like `"1e9"` parses as finite but produces a multi-decade delay that would
 * wedge the retry loop, so we clamp here.
 */
const MAX_RETRY_AFTER_MS = 5 * 60 * 1000;

/**
 * Parse an HTTP `Retry-After` header into milliseconds-from-now. Accepts both
 * the numeric seconds form ("120") and the HTTP-date form
 * ("Wed, 21 Oct 2015 07:28:00 GMT"). Returns `undefined` when the header is
 * missing or unparseable. The result is clamped to `[0, MAX_RETRY_AFTER_MS]`.
 */
export function parseRetryAfter(header: string | null): number | undefined {
  if (header == null) return undefined;
  const trimmed = header.trim();
  if (trimmed === "") return undefined;
  const seconds = Number(trimmed);
  if (Number.isFinite(seconds)) {
    return clampRetryAfter(seconds * 1000);
  }
  const date = Date.parse(trimmed);
  if (!Number.isNaN(date)) {
    return clampRetryAfter(date - Date.now());
  }
  return undefined;
}

function clampRetryAfter(ms: number): number {
  return Math.min(MAX_RETRY_AFTER_MS, Math.max(0, ms));
}

/**
 * AIMD (additive-increase / multiplicative-decrease) controller for in-flight
 * concurrency. The cap starts at `initial`, halves on each `onRateLimited()`
 * (floored at `min`), and recovers by 1 after `RECOVERY_THRESHOLD` consecutive
 * `onSuccess()` calls (capped at `initial`).
 */
export interface AIMDController {
  limit(): number;
  onSuccess(): void;
  onRateLimited(): void;
}

const RECOVERY_THRESHOLD = 8;

export function createAIMDController(options: { initial: number; min?: number }): AIMDController {
  const initial = Math.max(1, options.initial);
  const min = Math.max(1, options.min ?? 1);
  let current = initial;
  let successes = 0;

  return {
    limit() {
      return current;
    },
    onSuccess() {
      if (current >= initial) {
        successes = 0;
        return;
      }
      successes++;
      if (successes >= RECOVERY_THRESHOLD) {
        current = Math.min(initial, current + 1);
        successes = 0;
      }
    },
    onRateLimited() {
      current = Math.max(min, Math.floor(current / 2));
      successes = 0;
    },
  };
}

/**
 * Run `fn`, retrying on errors that `onError` accepts. `onError` decides per
 * attempt: returning `{ delayMs }` triggers a sleep and another attempt;
 * returning `null` rethrows immediately. Bounded by `maxRetries`; once
 * exhausted the last error is rethrown.
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    onError: (err: unknown, attempt: number) => { delayMs: number } | null;
  },
): Promise<T> {
  const maxRetries = options.maxRetries ?? 5;
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err) {
      if (attempt >= maxRetries) {
        throw err;
      }
      const decision = options.onError(err, attempt);
      if (decision == null) {
        throw err;
      }
      attempt++;
      if (decision.delayMs > 0) {
        await delay(decision.delayMs);
      }
    }
  }
}

/**
 * Full-jitter exponential backoff: `base * 2^attempt` is the cap, the actual
 * delay is uniformly random in `[base, cap]`. Caller-supplied `random` keeps
 * the function pure for tests; defaults to `Math.random`.
 */
export function expBackoff(attempt: number, baseMs: number, random: () => number = Math.random): number {
  const cap = baseMs * Math.pow(2, attempt);
  return baseMs + (cap - baseMs) * random();
}
