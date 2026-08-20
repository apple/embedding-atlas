// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { beforeEach, describe, expect, it } from "vitest";

import { SearchIndex } from "../src/search/search-index.js";

describe("SearchIndex", () => {
  let index: SearchIndex;

  beforeEach(() => {
    index = new SearchIndex();
    index.addPoints([
      { id: 1, text: "ALDI" },
      { id: 2, text: "ALDEA HOMES" },
      { id: 3, text: "ALDI Grocery Store" },
      { id: 4, text: "Aldente Pasta Bar" },
      { id: 5, text: "Trader Joe's" },
      { id: 6, text: "Café Restaurant" },
      { id: 7, text: "Walgreens Pharmacy" },
    ]);
  });

  it("matches an exact substring", () => {
    const results = index.query("aldi", 10);
    expect(results).toContain(1);
    expect(results).toContain(3);
  });

  it("does not return unrelated words that merely share a prefix (regression for #137)", () => {
    const results = index.query("aldi", 10);
    expect(results).not.toContain(2); // "ALDEA HOMES"
    expect(results).not.toContain(4); // "Aldente Pasta Bar"
  });

  it("is case-insensitive", () => {
    const results = index.query("ALDI", 10);
    expect(results).toContain(1);
  });

  it("still tolerates prefix queries (forward tokenization preserved)", () => {
    const results = index.query("walgreen", 10);
    expect(results).toContain(7);
  });

  it("still normalizes diacritics", () => {
    const results = index.query("cafe", 10);
    expect(results).toContain(6);
  });

  it("returns an empty array for queries with no match", () => {
    const results = index.query("zzzzzz", 10);
    expect(results).toEqual([]);
  });

  it("respects the limit parameter", () => {
    const results = index.query("aldi", 1);
    expect(results.length).toBe(1);
  });

  it("clear() removes previously indexed points", () => {
    index.clear();
    index.addPoints([{ id: 100, text: "Only This" }]);
    expect(index.query("aldi", 10)).toEqual([]);
    expect(index.query("only", 10)).toContain(100);
  });
});
