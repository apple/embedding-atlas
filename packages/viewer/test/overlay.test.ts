// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";
import { FocusRequestTracker, resolveEffectiveOverlay } from "../src/utils/overlay";
import type { Overlay } from "../src/charts/chart";

describe("resolveEffectiveOverlay", () => {
  const internal: Overlay = { nodes: ["internal"] };
  const external: Overlay = { nodes: ["A", "B", "C"] };

  it("uses the internal overlay when the external overlay is undefined", () => {
    expect(resolveEffectiveOverlay(undefined, internal)).toBe(internal);
  });

  it("uses the internal overlay when the external overlay is null", () => {
    expect(resolveEffectiveOverlay(null, internal)).toBe(internal);
  });

  it("uses the external overlay when one is provided", () => {
    expect(resolveEffectiveOverlay(external, internal)).toBe(external);
  });

  it("uses the external overlay even when the internal overlay is null", () => {
    expect(resolveEffectiveOverlay(external, null)).toBe(external);
  });

  it("an explicitly empty external overlay wins over the internal overlay", () => {
    const empty: Overlay = { nodes: [] };
    expect(resolveEffectiveOverlay(empty, internal)).toBe(empty);
  });

  it("returns null when neither overlay is set", () => {
    expect(resolveEffectiveOverlay(undefined, null)).toBe(null);
    expect(resolveEffectiveOverlay(null, null)).toBe(null);
  });
});

describe("FocusRequestTracker", () => {
  it("ignores a null request", () => {
    const tracker = new FocusRequestTracker();
    expect(tracker.accept(null)).toBe(null);
  });

  it("accepts the first request", () => {
    const tracker = new FocusRequestTracker();
    expect(tracker.accept({ id: "A", key: "run-1" })).not.toBe(null);
  });

  it("ignores a repeated request with the same key", () => {
    const tracker = new FocusRequestTracker();
    expect(tracker.accept({ id: "A", key: "run-1" })).not.toBe(null);
    expect(tracker.accept({ id: "A", key: "run-1" })).toBe(null);
  });

  it("accepts the same point again with a new key", () => {
    const tracker = new FocusRequestTracker();
    expect(tracker.accept({ id: "A", key: "run-1" })).not.toBe(null);
    expect(tracker.accept({ id: "A", key: "run-2" })).not.toBe(null);
  });

  it("accepts a different point with a new key", () => {
    const tracker = new FocusRequestTracker();
    expect(tracker.accept({ id: "A", key: 1 })).not.toBe(null);
    expect(tracker.accept({ id: "B", key: 2 })).not.toBe(null);
  });

  it("treats numeric key 0 as a valid key", () => {
    const tracker = new FocusRequestTracker();
    expect(tracker.accept({ id: 0, key: 0 })).not.toBe(null);
    expect(tracker.accept({ id: 0, key: 0 })).toBe(null);
    expect(tracker.accept({ id: 0, key: 1 })).not.toBe(null);
  });

  it("distinguishes string and numeric keys", () => {
    const tracker = new FocusRequestTracker();
    expect(tracker.accept({ id: "A", key: 1 })).not.toBe(null);
    expect(tracker.accept({ id: "A", key: "1" })).not.toBe(null);
  });

  it("only the latest accepted request stays latest", () => {
    const tracker = new FocusRequestTracker();
    const first = tracker.accept({ id: "A", key: "run-1" })!;
    const second = tracker.accept({ id: "B", key: "run-2" })!;
    expect(tracker.isLatest(first)).toBe(false);
    expect(tracker.isLatest(second)).toBe(true);
  });

  it("an ignored request does not invalidate the latest accepted one", () => {
    const tracker = new FocusRequestTracker();
    const seq = tracker.accept({ id: "A", key: "run-1" })!;
    expect(tracker.accept({ id: "A", key: "run-1" })).toBe(null);
    expect(tracker.accept(null)).toBe(null);
    expect(tracker.isLatest(seq)).toBe(true);
  });
});
