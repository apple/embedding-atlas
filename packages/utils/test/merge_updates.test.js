// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { applyUpdatesForKeyIfNeeded, applyUpdatesIfNeeded, mergeUpdates } from "@embedding-atlas/utils";

import { describe, expect, it } from "vitest";

describe("mergeUpdates", () => {
  describe("basic updates", () => {
    it("should return undefined when no updates are needed", () => {
      expect(mergeUpdates({ a: 1 }, {})).toBeUndefined();
      expect(mergeUpdates({ a: 1 }, { a: 1 })).toBeUndefined();
    });

    it("should update a single property", () => {
      expect(mergeUpdates({ a: 1, b: 2 }, { a: 3 })).toEqual({ a: 3, b: 2 });
    });

    it("should update multiple properties", () => {
      expect(mergeUpdates({ a: 1, b: 2, c: 3 }, { a: 10, c: 30 })).toEqual({
        a: 10,
        b: 2,
        c: 30,
      });
    });

    it("should remove property when update is undefined", () => {
      expect(mergeUpdates({ a: 1, b: 2 }, { a: undefined })).toEqual({ b: 2 });
      expect(mergeUpdates({ a: 1, b: 2, c: 3 }, { a: undefined, c: undefined })).toEqual({
        b: 2,
      });
    });

    it("should handle adding new properties", () => {
      expect(mergeUpdates({ a: 1 }, { b: 2 })).toEqual({ a: 1, b: 2 });
    });
  });

  describe("nested objects", () => {
    it("should merge nested objects recursively", () => {
      const value = { a: 1, b: { c: 2, d: 3 } };
      const updates = { b: { c: 4 } };
      expect(mergeUpdates(value, updates)).toEqual({ a: 1, b: { c: 4, d: 3 } });
    });

    it("should return undefined for nested objects with no changes", () => {
      const value = { a: 1, b: { c: 2, d: 3 } };
      const updates = { b: { c: 2 } };
      expect(mergeUpdates(value, updates)).toBeUndefined();
    });

    it("should handle deeply nested updates", () => {
      const value = { a: { b: { c: { d: 1 } } } };
      const updates = { a: { b: { c: { d: 2 } } } };
      expect(mergeUpdates(value, updates)).toEqual({ a: { b: { c: { d: 2 } } } });
    });

    it("should remove nested properties", () => {
      const value = { a: 1, b: { c: 2, d: 3 } };
      const updates = { b: { c: undefined } };
      expect(mergeUpdates(value, updates)).toEqual({ a: 1, b: { d: 3 } });
    });

    it("should handle combined nested update example from docs", () => {
      const value = { a: 1, b: { c: 2, d: 3 } };
      const updates = { b: { c: 4 }, a: undefined };
      expect(mergeUpdates(value, updates)).toEqual({ b: { c: 4, d: 3 } });
    });
  });

  describe("arrays", () => {
    it("should replace arrays atomically", () => {
      expect(mergeUpdates({ a: [1, 2, 3] }, { a: [4, 5, 6] })).toEqual({ a: [4, 5, 6] });
    });

    it("should return undefined when arrays are equal", () => {
      expect(mergeUpdates({ a: [1, 2, 3] }, { a: [1, 2, 3] })).toBeUndefined();
    });

    it("should handle nested arrays in objects", () => {
      expect(mergeUpdates({ a: { b: [1, 2] } }, { a: { b: [3, 4] } })).toEqual({
        a: { b: [3, 4] },
      });
    });
  });

  describe("primitive values", () => {
    it("should replace primitive values", () => {
      expect(mergeUpdates(1, 2)).toBe(2);
      expect(mergeUpdates("hello", "world")).toBe("world");
      expect(mergeUpdates(true, false)).toBe(false);
    });

    it("should return undefined for equal primitive values", () => {
      expect(mergeUpdates(1, 1)).toBeUndefined();
      expect(mergeUpdates("hello", "hello")).toBeUndefined();
      expect(mergeUpdates(true, true)).toBeUndefined();
    });
  });

  describe("immutability", () => {
    it("should not mutate the original value", () => {
      const value = { a: 1, b: { c: 2 } };
      const updates = { b: { c: 3 } };
      const valueCopy = JSON.parse(JSON.stringify(value));

      mergeUpdates(value, updates);

      expect(value).toEqual(valueCopy);
    });

    it("should not mutate the updates object", () => {
      const value = { a: 1, b: { c: 2 } };
      const updates = { b: { c: 3 } };
      const updatesCopy = JSON.parse(JSON.stringify(updates));

      mergeUpdates(value, updates);

      expect(updates).toEqual(updatesCopy);
    });

    it("should return a new object reference when changes are made", () => {
      const value = { a: 1, b: { c: 2 } };
      const result = mergeUpdates(value, { a: 2 });

      expect(result).not.toBe(value);
    });
  });

  describe("edge cases", () => {
    it("should handle empty objects", () => {
      expect(mergeUpdates({}, {})).toBeUndefined();
    });

    it("should handle null values", () => {
      expect(mergeUpdates({ a: null }, { a: null })).toBeUndefined();
      expect(mergeUpdates({ a: 1 }, { a: null })).toEqual({ a: null });
    });

    it("should handle mixed types", () => {
      expect(mergeUpdates({ a: { b: 1 } }, { a: [1, 2, 3] })).toEqual({ a: [1, 2, 3] });
      expect(mergeUpdates({ a: [1, 2, 3] }, { a: { b: 1 } })).toEqual({ a: { b: 1 } });
    });
  });
});

describe("applyUpdatesIfNeeded", () => {
  describe("merge mode", () => {
    it("should call setter when merge produces changes", () => {
      const value = { a: 1, b: 2 };
      const updates = { a: 3 };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesIfNeeded(value, updates, "merge", setter);

      expect(result).toEqual({ result: { a: 3, b: 2 } });
    });

    it("should return undefined when merge produces no changes", () => {
      const value = { a: 1, b: 2 };
      const updates = { a: 1 };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesIfNeeded(value, updates, "merge", setter);

      expect(result).toBeUndefined();
    });

    it("should handle nested merges", () => {
      const value = { a: { b: 1, c: 2 } };
      const updates = { a: { b: 3 } };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesIfNeeded(value, updates, "merge", setter);

      expect(result).toEqual({ result: { a: { b: 3, c: 2 } } });
    });
  });

  describe("replace mode", () => {
    it("should call setter when values differ", () => {
      const value = { a: 1 };
      const updates = { a: 2 };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesIfNeeded(value, updates, "replace", setter);

      expect(result).toEqual({ result: { a: 2 } });
    });

    it("should return undefined when values are identical", () => {
      const value = { a: 1 };
      const updates = value;
      const setter = (v) => ({ result: v });

      const result = applyUpdatesIfNeeded(value, updates, "replace", setter);

      expect(result).toBeUndefined();
    });

    it("should replace entire object even with partial updates", () => {
      const value = { a: 1, b: 2 };
      const updates = { a: 1 };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesIfNeeded(value, updates, "replace", setter);

      expect(result).toEqual({ result: { a: 1 } });
    });
  });
});

describe("applyUpdatesForKeyIfNeeded", () => {
  describe("merge mode", () => {
    it("should update a single key in records", () => {
      const records = { key1: { a: 1 }, key2: { a: 2 } };
      const update = { a: 3 };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesForKeyIfNeeded(records, "key1", update, "merge", setter);

      expect(result).toEqual({
        result: { key1: { a: 3 }, key2: { a: 2 } },
      });
    });

    it("should return undefined when no changes needed", () => {
      const records = { key1: { a: 1 }, key2: { a: 2 } };
      const update = { a: 1 };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesForKeyIfNeeded(records, "key1", update, "merge", setter);

      expect(result).toBeUndefined();
    });

    it("should merge nested updates for a key", () => {
      const records = { key1: { a: { b: 1, c: 2 } } };
      const update = { a: { b: 3 } };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesForKeyIfNeeded(records, "key1", update, "merge", setter);

      expect(result).toEqual({
        result: { key1: { a: { b: 3, c: 2 } } },
      });
    });

    it("should handle removing properties via undefined", () => {
      const records = { key1: { a: 1, b: 2 } };
      const update = { a: undefined };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesForKeyIfNeeded(records, "key1", update, "merge", setter);

      expect(result).toEqual({
        result: { key1: { b: 2 } },
      });
    });
  });

  describe("replace mode", () => {
    it("should replace value for a key", () => {
      const records = { key1: { a: 1 }, key2: { a: 2 } };
      const update = { a: 3 };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesForKeyIfNeeded(records, "key1", update, "replace", setter);

      expect(result).toEqual({
        result: { key1: { a: 3 }, key2: { a: 2 } },
      });
    });

    it("should return undefined when value is identical", () => {
      const value = { a: 1 };
      const records = { key1: value };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesForKeyIfNeeded(records, "key1", value, "replace", setter);

      expect(result).toBeUndefined();
    });

    it("should replace entire value for key", () => {
      const records = { key1: { a: 1, b: 2 } };
      const update = { a: 1 };
      const setter = (v) => ({ result: v });

      const result = applyUpdatesForKeyIfNeeded(records, "key1", update, "replace", setter);

      expect(result).toEqual({
        result: { key1: { a: 1 } },
      });
    });

    it("should not mutate original records", () => {
      const records = { key1: { a: 1 }, key2: { a: 2 } };
      const recordsCopy = JSON.parse(JSON.stringify(records));
      const update = { a: 3 };
      const setter = (v) => ({ result: v });

      applyUpdatesForKeyIfNeeded(records, "key1", update, "replace", setter);

      expect(records).toEqual(recordsCopy);
    });
  });
});
