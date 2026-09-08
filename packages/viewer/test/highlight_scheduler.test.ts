// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it, vi } from "vitest";

import { queryKeyOf, schedule, type HighlightSpan } from "../src/highlight/scheduler.js";

const span = (start: number, end: number, value: number): HighlightSpan => ({ start, end, value });

describe("queryKeyOf", () => {
  it("maps nullish to empty string", () => {
    expect(queryKeyOf(undefined)).toBe("");
    expect(queryKeyOf(null)).toBe("");
  });

  it("gives equal primitives equal keys and distinguishes types", () => {
    expect(queryKeyOf("a")).toBe(queryKeyOf("a"));
    expect(queryKeyOf(5)).toBe(queryKeyOf(5));
    expect(queryKeyOf("5")).not.toBe(queryKeyOf(5));
  });

  it("keys objects by reference", () => {
    const a = { q: 1 };
    const b = { q: 1 };
    expect(queryKeyOf(a)).toBe(queryKeyOf(a));
    expect(queryKeyOf(a)).not.toBe(queryKeyOf(b));
  });
});

describe("schedule", () => {
  it("coalesces a batch into one scorer call and aligns results by text", async () => {
    const scorer = vi.fn(async (texts: string[], _query: unknown) => texts.map((t, i) => [span(0, t.length, i)]));

    const [r1, r2, r3] = await Promise.all([
      schedule(scorer, "q", "needle", "alpha", 1),
      schedule(scorer, "q", "needle", "beta", 1),
      schedule(scorer, "q", "needle", "alpha", 1), // duplicate text
    ]);

    expect(scorer).toHaveBeenCalledTimes(1);
    expect(scorer.mock.calls[0][0]).toEqual(["alpha", "beta"]); // deduped, insertion order
    expect(scorer.mock.calls[0][1]).toBe("needle"); // query forwarded

    // "alpha" is index 0, "beta" is index 1 -> value mirrors index.
    expect(r1).toEqual([span(0, 5, 0)]);
    expect(r3).toEqual([span(0, 5, 0)]); // both "alpha" waiters get the same result
    expect(r2).toEqual([span(0, 4, 1)]);
  });

  it("separates batches by query key", async () => {
    const scorer = vi.fn(async (texts: string[]) => texts.map(() => []));
    await Promise.all([schedule(scorer, "q1", "a", "x", 1), schedule(scorer, "q2", "b", "x", 1)]);
    expect(scorer).toHaveBeenCalledTimes(2);
  });

  it("rejects and drops an aborted waiter; an empty batch never calls the scorer", async () => {
    const scorer = vi.fn(async (texts: string[]) => texts.map(() => []));
    const controller = new AbortController();
    const pending = schedule(scorer, "q", "needle", "only", 5, controller.signal);
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(scorer).not.toHaveBeenCalled();
  });

  it("still scores remaining texts when one waiter aborts", async () => {
    const scorer = vi.fn(async (texts: string[]) => texts.map((t) => [span(0, t.length, 1)]));
    const controller = new AbortController();
    const aborted = schedule(scorer, "q", "needle", "drop", 5, controller.signal);
    const kept = schedule(scorer, "q", "needle", "keep", 5);
    controller.abort();

    await expect(aborted).rejects.toMatchObject({ name: "AbortError" });
    await expect(kept).resolves.toEqual([span(0, 4, 1)]);
    expect(scorer).toHaveBeenCalledTimes(1);
    expect(scorer.mock.calls[0][0]).toEqual(["keep"]);
  });

  it("rejects all waiters when the scorer throws", async () => {
    const scorer = vi.fn(async () => {
      throw new Error("boom");
    });
    const a = schedule(scorer, "q", "needle", "a", 1);
    const b = schedule(scorer, "q", "needle", "b", 1);
    await expect(a).rejects.toThrow("boom");
    await expect(b).rejects.toThrow("boom");
  });

  it("rejects all waiters when the scorer throws synchronously", async () => {
    // A non-async scorer that throws before returning a promise must not leave
    // waiters unsettled (which would hang Promise.all in the action).
    const scorer = (() => {
      throw new Error("sync boom");
    }) as unknown as Parameters<typeof schedule>[0];
    const a = schedule(scorer, "q", "needle", "a", 1);
    const b = schedule(scorer, "q", "needle", "b", 1);
    await expect(a).rejects.toThrow("sync boom");
    await expect(b).rejects.toThrow("sync boom");
  });

  it("rejects all waiters when the scorer returns fewer results than texts", async () => {
    // Contract violation: one span array per text. A short result must surface as
    // an error, not silently resolve trailing texts to empty highlights.
    const scorer = vi.fn(async (texts: string[]) => texts.slice(1).map((t) => [span(0, t.length, 1)]));
    const a = schedule(scorer, "q", "needle", "a", 1);
    const b = schedule(scorer, "q", "needle", "b", 1);
    await expect(a).rejects.toThrow(/span arrays/);
    await expect(b).rejects.toThrow(/span arrays/);
  });
});
