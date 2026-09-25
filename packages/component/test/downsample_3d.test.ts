// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { pickVisibleIndices, type FrustumLike } from "../src/lib/embedding_view/downsample_3d.js";

const alwaysVisible: FrustumLike = { containsPoint: () => true };
const neverVisible: FrustumLike = { containsPoint: () => false };

/** Build interleaved x,y,z positions where point `i` is at (i, 0, 0). */
function makePositions(count: number): Float32Array {
  let positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) positions[i * 3] = i;
  return positions;
}

/** Run `pickVisibleIndices` to completion under fake timers (the module falls back to
 * `setTimeout`-based idle scheduling in Node, where `requestIdleCallback` doesn't exist) and
 * resolve with whatever `onComplete` received, or `null` if it was never called. */
async function runToCompletion(
  options: Omit<Parameters<typeof pickVisibleIndices>[0], "onComplete">,
): Promise<Uint32Array | null> {
  let result: Uint32Array | null = null;
  pickVisibleIndices({ ...options, onComplete: (indices) => (result = indices) });
  await vi.runAllTimersAsync();
  return result;
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("pickVisibleIndices", () => {
  it("excludes points the frustum rejects", async () => {
    // Points 0..9 at x=0..9; a frustum that only accepts x >= 5.
    let positions = makePositions(10);
    let frustum: FrustumLike = { containsPoint: (p) => p.x >= 5 };
    let result = await runToCompletion({ positions, count: 10, frustum, maxPoints: 100, isStale: () => false });
    expect(Array.from(result!)).toEqual([5, 6, 7, 8, 9]);
  });

  it("returns nothing when the frustum rejects everything", async () => {
    let positions = makePositions(10);
    let result = await runToCompletion({
      positions,
      count: 10,
      frustum: neverVisible,
      maxPoints: 100,
      isStale: () => false,
    });
    expect(result).toHaveLength(0);
  });

  it("caps the result at maxPoints even when far more points are visible", async () => {
    let count = 1000;
    let positions = makePositions(count);
    let result = await runToCompletion({
      positions,
      count,
      frustum: alwaysVisible,
      maxPoints: 100,
      isStale: () => false,
    });
    // Reservoir sampling makes *which* 100 indices random, but the count is always exact.
    expect(result).toHaveLength(100);
    for (let index of result!) {
      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThan(count);
    }
  });

  it("returns every index when maxPoints exceeds the visible count", async () => {
    let count = 50;
    let positions = makePositions(count);
    let result = await runToCompletion({
      positions,
      count,
      frustum: alwaysVisible,
      maxPoints: 1000,
      isStale: () => false,
    });
    expect(Array.from(result!)).toEqual(Array.from({ length: count }, (_, i) => i));
  });

  it("processes datasets spanning multiple idle-scheduled chunks", async () => {
    // CHUNK_SIZE is 20000, so this spans three idle callbacks (20000 + 20000 + 5000).
    let count = 45000;
    let positions = makePositions(count);
    let result = await runToCompletion({
      positions,
      count,
      frustum: alwaysVisible,
      maxPoints: count,
      isStale: () => false,
    });
    expect(result).toHaveLength(count);
    expect(Array.from(result!)).toEqual(Array.from({ length: count }, (_, i) => i));
  });

  it("completes immediately with an empty result for an empty dataset", async () => {
    let result = await runToCompletion({
      positions: new Float32Array(0),
      count: 0,
      frustum: alwaysVisible,
      maxPoints: 100,
      isStale: () => false,
    });
    expect(result).not.toBeNull();
    expect(result).toHaveLength(0);
  });

  it("stops without calling onComplete once isStale becomes true", async () => {
    // Large enough to require a second idle-scheduled chunk; isStale flips to true after the
    // first chunk's checks, so the second chunk should bail before ever finishing.
    let count = 45000;
    let positions = makePositions(count);
    let calls = 0;
    let isStale = () => {
      calls++;
      return calls > 2; // let the first chunk's start/end checks through, then go stale
    };
    let result = await runToCompletion({ positions, count, frustum: alwaysVisible, maxPoints: count, isStale });
    expect(result).toBeNull();
  });
});
