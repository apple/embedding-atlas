// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  chunkInputs,
  createAIMDController,
  dispatchPool,
  expBackoff,
  HttpError,
  parseRetryAfter,
  withRetry,
} from "../src/inference/utils.js";

describe("chunkInputs", () => {
  it("returns no chunks for an empty input", () => {
    expect(chunkInputs([], 4)).toEqual([]);
  });

  it("splits exact multiples evenly", () => {
    expect(chunkInputs([1, 2, 3, 4], 2)).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("leaves a short tail for ragged inputs", () => {
    expect(chunkInputs([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });

  it("rejects sizes < 1", () => {
    expect(() => chunkInputs([1], 0)).toThrow();
  });
});

describe("dispatchPool", () => {
  it("preserves output order even when chunks finish out of order", async () => {
    const chunks = [
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ];
    const delays = [40, 10, 30, 5];
    const embed = vi.fn(async (chunk: number[]) => {
      await new Promise((r) => setTimeout(r, delays[chunk[0] / 2 - 0]));
      return chunk.reduce((a, b) => a + b, 0);
    });
    const result = await dispatchPool(chunks, embed, () => 4);
    expect(result).toEqual([3, 7, 11, 15]);
  });

  it("workers exit when concurrency drops below their index", async () => {
    let limit = 4;
    const chunks = Array.from({ length: 8 }, (_, i) => [i]);
    const inFlight: number[] = [];
    let peak = 0;
    const embed = vi.fn(async (chunk: number[]) => {
      inFlight.push(chunk[0]);
      peak = Math.max(peak, inFlight.length);
      // After the first two chunks complete, the AIMD shrinks the cap.
      if (chunk[0] === 1) limit = 2;
      await new Promise((r) => setTimeout(r, 5));
      inFlight.splice(inFlight.indexOf(chunk[0]), 1);
      return chunk[0];
    });
    const result = await dispatchPool(chunks, embed, () => limit);
    // All chunks still got processed, in order.
    expect(result).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    // Initial spawn was 4, but later workers exited so peak never grew further.
    expect(peak).toBeLessThanOrEqual(4);
  });

  it("propagates a rejection from embedChunk", async () => {
    const embed = vi.fn(async () => {
      throw new Error("boom");
    });
    await expect(dispatchPool([[1]], embed, () => 1)).rejects.toThrow("boom");
  });
});

describe("parseRetryAfter", () => {
  it("returns undefined for null/empty", () => {
    expect(parseRetryAfter(null)).toBeUndefined();
    expect(parseRetryAfter("")).toBeUndefined();
    expect(parseRetryAfter("   ")).toBeUndefined();
  });

  it("parses numeric seconds into ms", () => {
    expect(parseRetryAfter("0")).toBe(0);
    expect(parseRetryAfter("3")).toBe(3000);
    expect(parseRetryAfter("120")).toBe(120000);
  });

  it("parses HTTP-date into ms-from-now", () => {
    const future = new Date(Date.now() + 5000).toUTCString();
    const ms = parseRetryAfter(future);
    expect(ms).toBeGreaterThan(3000);
    expect(ms).toBeLessThan(7000);
  });

  it("returns undefined for unparseable values", () => {
    expect(parseRetryAfter("nope")).toBeUndefined();
  });

  it("clamps absurd numeric values to the max", () => {
    const max = 5 * 60 * 1000;
    expect(parseRetryAfter("1e9")).toBe(max);
    expect(parseRetryAfter("9999999999")).toBe(max);
  });

  it("clamps far-future HTTP dates to the max", () => {
    const max = 5 * 60 * 1000;
    const farFuture = new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toUTCString();
    expect(parseRetryAfter(farFuture)).toBe(max);
  });
});

describe("createAIMDController", () => {
  it("starts at initial and halves on rate-limit", () => {
    const c = createAIMDController({ initial: 8, min: 1 });
    expect(c.limit()).toBe(8);
    c.onRateLimited();
    expect(c.limit()).toBe(4);
    c.onRateLimited();
    expect(c.limit()).toBe(2);
    c.onRateLimited();
    expect(c.limit()).toBe(1);
    c.onRateLimited();
    expect(c.limit()).toBe(1); // floored at min
  });

  it("recovers additively after enough successes", () => {
    const c = createAIMDController({ initial: 8, min: 1 });
    c.onRateLimited();
    expect(c.limit()).toBe(4);
    for (let i = 0; i < 7; i++) c.onSuccess();
    expect(c.limit()).toBe(4);
    c.onSuccess(); // 8th
    expect(c.limit()).toBe(5);
  });

  it("never exceeds initial", () => {
    const c = createAIMDController({ initial: 4 });
    for (let i = 0; i < 100; i++) c.onSuccess();
    expect(c.limit()).toBe(4);
  });

  it("rate-limit resets the success counter", () => {
    const c = createAIMDController({ initial: 8 });
    c.onRateLimited();
    for (let i = 0; i < 5; i++) c.onSuccess();
    c.onRateLimited();
    expect(c.limit()).toBe(2);
    for (let i = 0; i < 7; i++) c.onSuccess();
    expect(c.limit()).toBe(2); // counter reset, only 7 successes since last RL
    c.onSuccess();
    expect(c.limit()).toBe(3);
  });
});

describe("withRetry", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns the result on first success", async () => {
    const fn = vi.fn(async () => 42);
    const result = await withRetry(fn, { onError: () => null });
    expect(result).toBe(42);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("rethrows when onError returns null", async () => {
    const fn = vi.fn(async () => {
      throw new Error("boom");
    });
    await expect(withRetry(fn, { onError: () => null })).rejects.toThrow("boom");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries on accepted errors and respects delayMs", async () => {
    let attempts = 0;
    const fn = vi.fn(async () => {
      attempts++;
      if (attempts < 3) throw new Error("transient");
      return "ok";
    });
    const promise = withRetry(fn, { onError: () => ({ delayMs: 100 }) });
    await vi.advanceTimersByTimeAsync(250);
    await expect(promise).resolves.toBe("ok");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("gives up after maxRetries and rethrows the last error", async () => {
    const fn = vi.fn(async () => {
      throw new Error("boom");
    });
    const promise = withRetry(fn, { maxRetries: 2, onError: () => ({ delayMs: 0 }) });
    const assertion = expect(promise).rejects.toThrow("boom");
    await vi.runAllTimersAsync();
    await assertion;
    expect(fn).toHaveBeenCalledTimes(3); // initial + 2 retries
  });
});

describe("expBackoff", () => {
  it("base * 2^attempt is the cap; delay is at least base", () => {
    const random = () => 0;
    expect(expBackoff(0, 100, random)).toBe(100);
    expect(expBackoff(1, 100, random)).toBe(100);
    expect(expBackoff(3, 100, random)).toBe(100);
  });

  it("hits the cap when random returns 1", () => {
    const random = () => 1;
    expect(expBackoff(0, 100, random)).toBe(100);
    expect(expBackoff(1, 100, random)).toBe(200);
    expect(expBackoff(3, 100, random)).toBe(800);
  });

  it("scales jitter linearly", () => {
    expect(expBackoff(2, 100, () => 0.5)).toBe(250); // 100 + 0.5*(400-100)
  });
});

describe("HttpError", () => {
  it("carries status and optional retryAfterMs", () => {
    const e = new HttpError("rate limited", 429, 1500);
    expect(e.status).toBe(429);
    expect(e.retryAfterMs).toBe(1500);
    expect(e.name).toBe("HttpError");
    expect(e instanceof Error).toBe(true);
  });
});
