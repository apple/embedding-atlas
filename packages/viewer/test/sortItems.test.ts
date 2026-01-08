// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, test } from "vitest";
import { sortItems, type SortableItem } from "../src/charts/basic/sortItems.js";

describe("sortItems", () => {
  // Helper to create test items
  function createItem(x: string, total: number, selected: number, isSpecial = false): SortableItem {
    return { x, total, selected, isSpecial };
  }

  describe("sorting regular items", () => {
    test("sorts by total descending by default", () => {
      let items = [
        createItem("a", 10, 5),
        createItem("b", 30, 15),
        createItem("c", 20, 10),
      ];
      let result = sortItems(items, -1, undefined, undefined);

      expect(result.map((i) => i.x)).toEqual(["b", "c", "a"]);
    });

    test("sorts by total ascending", () => {
      let items = [
        createItem("a", 10, 5),
        createItem("b", 30, 15),
        createItem("c", 20, 10),
      ];
      let result = sortItems(items, -1, "total", "asc");

      expect(result.map((i) => i.x)).toEqual(["a", "c", "b"]);
    });

    test("sorts by selected descending", () => {
      let items = [
        createItem("a", 10, 5),
        createItem("b", 30, 3),
        createItem("c", 20, 10),
      ];
      let result = sortItems(items, -1, "selected", "desc");

      expect(result.map((i) => i.x)).toEqual(["c", "a", "b"]);
    });

    test("sorts by selected ascending", () => {
      let items = [
        createItem("a", 10, 5),
        createItem("b", 30, 3),
        createItem("c", 20, 10),
      ];
      let result = sortItems(items, -1, "selected", "asc");

      expect(result.map((i) => i.x)).toEqual(["b", "a", "c"]);
    });
  });

  describe("handling special items", () => {
    test("keeps special items at the end after sorting", () => {
      let items = [
        createItem("a", 10, 5),
        createItem("b", 30, 15),
        createItem("c", 20, 10),
        createItem("(null)", 5, 2, true),
        createItem("(5 others)", 8, 4, true),
      ];
      let result = sortItems(items, 3, "total", "desc");

      expect(result.map((i) => i.x)).toEqual(["b", "c", "a", "(null)", "(5 others)"]);
    });

    test("does not sort special items", () => {
      let items = [
        createItem("a", 10, 5),
        createItem("(5 others)", 100, 50, true),
        createItem("(null)", 5, 2, true),
      ];
      let result = sortItems(items, 1, "total", "desc");

      // Special items stay in their original order even though (5 others) has higher total
      expect(result.map((i) => i.x)).toEqual(["a", "(5 others)", "(null)"]);
    });

    test("handles array with only special items", () => {
      let items = [
        createItem("(null)", 5, 2, true),
        createItem("(5 others)", 8, 4, true),
      ];
      let result = sortItems(items, 0, "total", "desc");

      expect(result.map((i) => i.x)).toEqual(["(null)", "(5 others)"]);
    });

    test("handles firstSpecialIndex = -1 (no special items)", () => {
      let items = [
        createItem("a", 10, 5),
        createItem("b", 30, 15),
        createItem("c", 20, 10),
      ];
      let result = sortItems(items, -1, "total", "desc");

      expect(result.map((i) => i.x)).toEqual(["b", "c", "a"]);
    });
  });

  describe("edge cases", () => {
    test("handles empty array", () => {
      let result = sortItems([], -1, "total", "desc");
      expect(result).toEqual([]);
    });

    test("handles single item", () => {
      let items = [createItem("a", 10, 5)];
      let result = sortItems(items, -1, "total", "desc");

      expect(result).toEqual(items);
    });

    test("handles items with equal values", () => {
      let items = [
        createItem("a", 10, 5),
        createItem("b", 10, 5),
        createItem("c", 10, 5),
      ];
      let result = sortItems(items, -1, "total", "desc");

      expect(result.length).toBe(3);
      expect(result.every((i) => i.total === 10)).toBe(true);
    });

    test("does not mutate original array", () => {
      let items = [
        createItem("a", 10, 5),
        createItem("b", 30, 15),
        createItem("c", 20, 10),
      ];
      let originalOrder = items.map((i) => i.x);
      sortItems(items, -1, "total", "desc");

      expect(items.map((i) => i.x)).toEqual(originalOrder);
    });
  });
});
