// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, test } from "vitest";

import { parseExactPhrase, SearchIndex } from "../src/search/search_index.js";

describe("parseExactPhrase", () => {
  test("returns the inner phrase for a quoted query", () => {
    expect(parseExactPhrase('"aldi"')).toBe("aldi");
    expect(parseExactPhrase('"new york"')).toBe("new york");
  });

  test("returns null for an unquoted query", () => {
    expect(parseExactPhrase("aldi")).toBe(null);
    expect(parseExactPhrase('aldi"')).toBe(null);
    expect(parseExactPhrase('"aldi')).toBe(null);
  });

  test("returns null for an empty quoted query", () => {
    expect(parseExactPhrase('""')).toBe(null);
    expect(parseExactPhrase('"')).toBe(null);
  });
});

describe("SearchIndex exact-phrase search", () => {
  // Mirrors the report in issue #137: the fuzzy encoder maps "aldi" and
  // "aldea" to the same tokens, so a plain search for "aldi" surfaces
  // "ALDEA HOMES" before the real "ALDI" rows.
  const points = [
    { id: 1, text: "ALDEA HOMES" },
    { id: 2, text: "ALDEA HOMES TWO" },
    { id: 3, text: "ALDI Supermarket" },
    { id: 4, text: "Corner ALDI" },
    { id: 5, text: "Walmart" },
  ];

  test("the fuzzy search reproduces the false matches from the issue", () => {
    let index = new SearchIndex();
    index.addPoints(points);
    let result = index.query("aldi", 100);
    // The fuzzy encoder returns the "ALDEA" rows alongside the real matches,
    // which is the behavior the quoted search is meant to avoid.
    expect(result).toContain(1);
    expect(result).toContain(2);
  });

  test("a quoted query matches only the exact substring, case-insensitively", () => {
    let index = new SearchIndex();
    index.addPoints(points);
    let result = index.query('"aldi"', 100);
    expect(new Set(result)).toEqual(new Set([3, 4]));
    expect(result).not.toContain(1);
    expect(result).not.toContain(2);
    expect(result).not.toContain(5);
  });

  test("a quoted query respects the limit", () => {
    let index = new SearchIndex();
    index.addPoints(points);
    let result = index.query('"aldi"', 1);
    expect(result.length).toBe(1);
  });

  test("clear resets both the fuzzy index and the exact-match texts", () => {
    let index = new SearchIndex();
    index.addPoints(points);
    index.clear();
    expect(index.query('"aldi"', 100)).toEqual([]);
  });
});
