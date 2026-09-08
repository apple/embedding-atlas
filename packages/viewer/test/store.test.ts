// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { derived, get, writable } from "svelte/store";
import { describe, expect, it } from "vitest";
import { isolatedWritable, stableDerived, stableWritable } from "../src/utils/store";

// Helpers to track subscriber calls.
function track<T>(store: { subscribe: (fn: (v: T) => void) => () => void }) {
  const calls: T[] = [];
  const unsub = store.subscribe((v) => calls.push(v));
  return { calls, unsub };
}

describe("vanilla svelte stores", () => {
  describe("writable", () => {
    it("does not notify when set to the same primitive", () => {
      const s = writable(1);
      const { calls, unsub } = track(s);
      s.set(1);
      s.set(1);
      expect(calls).toEqual([1]);
      unsub();
    });

    it("notifies when set to the same object reference", () => {
      const obj = { a: 1 };
      const s = writable(obj);
      const { calls, unsub } = track(s);
      s.set(obj);
      s.set(obj);
      // safe_not_equal treats objects as always changed
      expect(calls).toEqual([obj, obj, obj]);
      unsub();
    });

    it("notifies when updated to the same object reference", () => {
      const obj = { a: 1 };
      const s = writable(obj);
      const { calls, unsub } = track(s);
      s.update((v) => v);
      expect(calls).toEqual([obj, obj]);
      unsub();
    });
  });

  describe("derived", () => {
    it("does not notify when derived primitive is unchanged", () => {
      const s = writable(1);
      const d = derived(s, (v) => v * 2);
      const { calls, unsub } = track(d);
      s.set(2);
      s.set(3);
      expect(calls).toEqual([2, 4, 6]);
      unsub();
    });

    it("notifies even when derived returns the same object reference", () => {
      const result = { x: 1 };
      const s = writable(1);
      const d = derived(s, () => result);
      const { calls, unsub } = track(d);
      s.set(2);
      s.set(3);
      // safe_not_equal treats objects as always changed
      expect(calls).toEqual([result, result, result]);
      unsub();
    });
  });
});

describe("stableWritable", () => {
  it("does not notify when set to the same primitive", () => {
    const s = stableWritable(1);
    const { calls, unsub } = track(s);
    s.set(1);
    s.set(1);
    expect(calls).toEqual([1]);
    unsub();
  });

  it("notifies when set to a different primitive", () => {
    const s = stableWritable(1);
    const { calls, unsub } = track(s);
    s.set(2);
    s.set(3);
    expect(calls).toEqual([1, 2, 3]);
    unsub();
  });

  it("does not notify when set to the same object reference", () => {
    const obj = { a: 1 };
    const s = stableWritable(obj);
    const { calls, unsub } = track(s);
    s.set(obj);
    s.set(obj);
    expect(calls).toEqual([obj]);
    unsub();
  });

  it("notifies when set to a different object with same contents", () => {
    const s = stableWritable({ a: 1 });
    const { calls, unsub } = track(s);
    const obj2 = { a: 1 };
    s.set(obj2);
    expect(calls).toEqual([{ a: 1 }, obj2]);
    unsub();
  });

  it("does not notify when updated to the same object reference", () => {
    const obj = { a: 1 };
    const s = stableWritable(obj);
    const { calls, unsub } = track(s);
    s.update((v) => v);
    s.update((v) => v);
    expect(calls).toEqual([obj]);
    unsub();
  });

  it("notifies when updated to a new object", () => {
    const s = stableWritable({ a: 1 });
    const { calls, unsub } = track(s);
    s.update((v) => ({ ...v, a: 2 }));
    expect(calls).toEqual([{ a: 1 }, { a: 2 }]);
    unsub();
  });

  it("works with get()", () => {
    const s = stableWritable(42);
    expect(get(s)).toBe(42);
    s.set(99);
    expect(get(s)).toBe(99);
  });
});

describe("stableDerived", () => {
  it("does not notify when derived primitive is unchanged", () => {
    const s = stableWritable(1);
    const d = stableDerived(s, (v) => v * 2);
    const { calls, unsub } = track(d);
    s.set(2);
    s.set(3);
    expect(calls).toEqual([2, 4, 6]);
    unsub();
  });

  it("does not notify when derived returns the same object reference", () => {
    const result = { x: 1 };
    const s = stableWritable(1);
    const d = stableDerived(s, () => result);
    const { calls, unsub } = track(d);
    s.set(2);
    s.set(3);
    expect(calls).toEqual([result]);
    unsub();
  });

  it("notifies when derived returns a new object", () => {
    const s = stableWritable(1);
    const d = stableDerived(s, (v) => ({ value: v }));
    const { calls, unsub } = track(d);
    s.set(2);
    expect(calls).toEqual([{ value: 1 }, { value: 2 }]);
    unsub();
  });

  it("skips when derived number is unchanged despite upstream change", () => {
    const s = stableWritable({ x: 1, y: 10 });
    const d = stableDerived(s, (v) => v.x);
    const { calls, unsub } = track(d);
    // Change y but not x
    s.set({ x: 1, y: 20 });
    s.set({ x: 1, y: 30 });
    expect(calls).toEqual([1]);
    unsub();
  });

  it("works with multiple input stores", () => {
    const a = stableWritable(1);
    const b = stableWritable(2);
    const d = stableDerived([a, b] as [typeof a, typeof b], ([va, vb]) => va + vb);
    const { calls, unsub } = track(d);
    a.set(3);
    b.set(4);
    expect(calls).toEqual([3, 5, 7]);
    unsub();
  });

  it("works with multiple input stores and skips unchanged", () => {
    const a = stableWritable(1);
    const b = stableWritable(2);
    const d = stableDerived([a, b] as [typeof a, typeof b], ([va, vb]) => va + vb);
    const { calls, unsub } = track(d);
    // 1+2=3, set a=2 -> 2+2=4, set b=1 -> 2+1=3, set a=1 -> 1+1=2
    a.set(2);
    b.set(1);
    a.set(1);
    expect(calls).toEqual([3, 4, 3, 2]);
    unsub();
  });

  it("works with vanilla writable as input", () => {
    const s = writable(1);
    const d = stableDerived(s, (v) => v * 2);
    const { calls, unsub } = track(d);
    s.set(2);
    expect(calls).toEqual([2, 4]);
    unsub();
  });

  it("works with get()", () => {
    const s = stableWritable(5);
    const d = stableDerived(s, (v) => v * 3);
    expect(get(d)).toBe(15);
    s.set(10);
    expect(get(d)).toBe(30);
  });
});

describe("isolatedWritable", () => {
  it("suppresses notifications from own set", () => {
    const inner = writable(1);
    const isolated = isolatedWritable(inner);
    const { calls, unsub } = track(isolated);
    isolated.set(2);
    isolated.set(3);
    expect(calls).toEqual([1]);
    unsub();
  });

  it("suppresses notifications from own update", () => {
    const inner = writable(1);
    const isolated = isolatedWritable(inner);
    const { calls, unsub } = track(isolated);
    isolated.update((v) => v + 1);
    isolated.update((v) => v + 1);
    expect(calls).toEqual([1]);
    unsub();
  });

  it("still notifies when the wrapped store is set directly", () => {
    const inner = writable(1);
    const isolated = isolatedWritable(inner);
    const { calls, unsub } = track(isolated);
    inner.set(2);
    inner.set(3);
    expect(calls).toEqual([1, 2, 3]);
    unsub();
  });

  it("actually writes through to the wrapped store", () => {
    const inner = writable(1);
    const isolated = isolatedWritable(inner);
    isolated.set(42);
    expect(get(inner)).toBe(42);
    isolated.update((v) => v + 8);
    expect(get(inner)).toBe(50);
  });

  it("resumes notifying after own write completes", () => {
    const inner = writable(1);
    const isolated = isolatedWritable(inner);
    const { calls, unsub } = track(isolated);
    isolated.set(2);
    // own write suppressed
    expect(calls).toEqual([1]);
    // external write goes through
    inner.set(3);
    expect(calls).toEqual([1, 3]);
    unsub();
  });
});
