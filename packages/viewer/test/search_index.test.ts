// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, test } from "vitest";

import { parseQuery, SearchIndex } from "../src/search/search_index.js";

describe("parseQuery", () => {
  test("an unquoted query is all free text with no phrases", () => {
    expect(parseQuery("aldi")).toEqual({ phrases: [], freeText: "aldi" });
    expect(parseQuery("aldi store")).toEqual({ phrases: [], freeText: "aldi store" });
  });

  test("a fully quoted query is a single phrase with no free text", () => {
    expect(parseQuery('"aldi"')).toEqual({ phrases: ["aldi"], freeText: "" });
    expect(parseQuery('"new york"')).toEqual({ phrases: ["new york"], freeText: "" });
  });

  test("a mixed query splits phrases from free text", () => {
    expect(parseQuery('"aldi" store')).toEqual({ phrases: ["aldi"], freeText: "store" });
    expect(parseQuery('store "aldi"')).toEqual({ phrases: ["aldi"], freeText: "store" });
    expect(parseQuery('"a" "b" c')).toEqual({ phrases: ["a", "b"], freeText: "c" });
  });

  test("empty quotes contribute no phrase", () => {
    expect(parseQuery('""')).toEqual({ phrases: [], freeText: "" });
    expect(parseQuery('"" store')).toEqual({ phrases: [], freeText: "store" });
  });

  test("an unterminated quote stays in the free text", () => {
    expect(parseQuery('"aldi')).toEqual({ phrases: [], freeText: '"aldi' });
    expect(parseQuery('store "aldi')).toEqual({ phrases: [], freeText: 'store "aldi' });
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
    { id: 6, text: "ALDI store downtown" },
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
    expect(new Set(result)).toEqual(new Set([3, 4, 6]));
    expect(result).not.toContain(1);
    expect(result).not.toContain(2);
    expect(result).not.toContain(5);
  });

  test("a mixed query requires the phrase and fuzzy-matches the free text", () => {
    let index = new SearchIndex();
    index.addPoints(points);
    // "aldi" must appear exactly, and "store" narrows via the fuzzy index, so
    // only the row that contains both survives.
    let result = index.query('"aldi" store', 100);
    expect(new Set(result)).toEqual(new Set([6]));
    expect(result).not.toContain(3);
    expect(result).not.toContain(4);
  });

  test("multiple phrases must all be present", () => {
    let index = new SearchIndex();
    index.addPoints(points);
    expect(new Set(index.query('"aldi" "store"', 100))).toEqual(new Set([6]));
    expect(index.query('"aldi" "walmart"', 100)).toEqual([]);
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
