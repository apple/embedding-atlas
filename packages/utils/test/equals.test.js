// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { deepEquals } from "@embedding-atlas/utils";

import { describe, expect, it } from "vitest";

describe("deepEquals", () => {
  describe("primitive values", () => {
    it("should return true for identical primitives", () => {
      expect(deepEquals(1, 1)).toBe(true);
      expect(deepEquals("test", "test")).toBe(true);
      expect(deepEquals(true, true)).toBe(true);
      expect(deepEquals(false, false)).toBe(true);
    });

    it("should return false for different primitives", () => {
      expect(deepEquals(1, 2)).toBe(false);
      expect(deepEquals("test", "other")).toBe(false);
      expect(deepEquals(true, false)).toBe(false);
    });

    it("should return false for different types", () => {
      expect(deepEquals(1, "1")).toBe(false);
      expect(deepEquals(0, false)).toBe(false);
      expect(deepEquals(null, undefined)).toBe(false);
    });
  });

  describe("null and undefined", () => {
    it("should return true for null === null", () => {
      expect(deepEquals(null, null)).toBe(true);
    });

    it("should return true for undefined === undefined", () => {
      expect(deepEquals(undefined, undefined)).toBe(true);
    });

    it("should return false for null vs object", () => {
      expect(deepEquals(null, {})).toBe(false);
      expect(deepEquals({}, null)).toBe(false);
    });

    it("should return false for undefined vs object", () => {
      expect(deepEquals(undefined, {})).toBe(false);
      expect(deepEquals({}, undefined)).toBe(false);
    });
  });

  describe("objects", () => {
    it("should return true for empty objects", () => {
      expect(deepEquals({}, {})).toBe(true);
    });

    it("should return true for identical simple objects", () => {
      expect(deepEquals({ a: 1 }, { a: 1 })).toBe(true);
      expect(deepEquals({ a: 1, b: 2 }, { a: 1, b: 2 })).toBe(true);
    });

    it("should return false for objects with different values", () => {
      expect(deepEquals({ a: 1 }, { a: 2 })).toBe(false);
    });

    it("should return false for objects with different keys", () => {
      expect(deepEquals({ a: 1 }, { b: 1 })).toBe(false);
    });

    it("should return false for objects with different number of keys", () => {
      expect(deepEquals({ a: 1 }, { a: 1, b: 2 })).toBe(false);
      expect(deepEquals({ a: 1, b: 2 }, { a: 1 })).toBe(false);
    });

    it("should return true for nested objects", () => {
      expect(deepEquals({ a: { b: 1 } }, { a: { b: 1 } })).toBe(true);
      expect(deepEquals({ a: { b: { c: 1 } } }, { a: { b: { c: 1 } } })).toBe(true);
    });

    it("should return false for different nested objects", () => {
      expect(deepEquals({ a: { b: 1 } }, { a: { b: 2 } })).toBe(false);
      expect(deepEquals({ a: { b: 1 } }, { a: { c: 1 } })).toBe(false);
    });

    it("should handle same reference", () => {
      const obj = { a: 1, b: 2 };
      expect(deepEquals(obj, obj)).toBe(true);
    });
  });

  describe("arrays", () => {
    it("should return true for empty arrays", () => {
      expect(deepEquals([], [])).toBe(true);
    });

    it("should return true for identical arrays", () => {
      expect(deepEquals([1, 2, 3], [1, 2, 3])).toBe(true);
    });

    it("should return false for arrays with different values", () => {
      expect(deepEquals([1, 2, 3], [1, 2, 4])).toBe(false);
    });

    it("should return false for arrays with different lengths", () => {
      expect(deepEquals([1, 2], [1, 2, 3])).toBe(false);
    });

    it("should return false for arrays with same elements in different order", () => {
      expect(deepEquals([1, 2, 3], [3, 2, 1])).toBe(false);
    });

    it("should return true for nested arrays", () => {
      expect(deepEquals([1, [2, 3]], [1, [2, 3]])).toBe(true);
      expect(
        deepEquals(
          [
            [1, 2],
            [3, 4],
          ],
          [
            [1, 2],
            [3, 4],
          ],
        ),
      ).toBe(true);
    });

    it("should return false for different nested arrays", () => {
      expect(deepEquals([1, [2, 3]], [1, [2, 4]])).toBe(false);
    });
  });

  describe("mixed structures", () => {
    it("should handle objects containing arrays", () => {
      expect(deepEquals({ a: [1, 2] }, { a: [1, 2] })).toBe(true);
      expect(deepEquals({ a: [1, 2] }, { a: [1, 3] })).toBe(false);
    });

    it("should handle arrays containing objects", () => {
      expect(deepEquals([{ a: 1 }], [{ a: 1 }])).toBe(true);
      expect(deepEquals([{ a: 1 }], [{ a: 2 }])).toBe(false);
    });

    it("should handle complex nested structures", () => {
      const obj1 = {
        a: 1,
        b: [2, 3, { c: 4 }],
        d: { e: { f: [5, 6] } },
      };
      const obj2 = {
        a: 1,
        b: [2, 3, { c: 4 }],
        d: { e: { f: [5, 6] } },
      };
      expect(deepEquals(obj1, obj2)).toBe(true);
    });

    it("should return false for complex structures with differences", () => {
      const obj1 = {
        a: 1,
        b: [2, 3, { c: 4 }],
        d: { e: { f: [5, 6] } },
      };
      const obj2 = {
        a: 1,
        b: [2, 3, { c: 4 }],
        d: { e: { f: [5, 7] } },
      };
      expect(deepEquals(obj1, obj2)).toBe(false);
    });
  });

  describe("edge cases", () => {
    it("should return false when comparing array with object", () => {
      expect(deepEquals([], {})).toBe(true); // Both have 0 keys
      expect(deepEquals([1], { 0: 1 })).toBe(true); // Array is object with numeric keys
    });

    it("should handle objects with null values", () => {
      expect(deepEquals({ a: null }, { a: null })).toBe(true);
      expect(deepEquals({ a: null }, { a: undefined })).toBe(false);
    });

    it("should handle objects with undefined values", () => {
      expect(deepEquals({ a: undefined }, { a: undefined })).toBe(true);
      expect(deepEquals({ a: undefined }, { a: null })).toBe(false);
    });
  });
});
