// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, test } from "vitest";

import { FullTextSearcher } from "../src/search/search.js";
import { escapeLikePattern, parseQuery } from "../src/search/query_parser.js";

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

describe("escapeLikePattern", () => {
  test("a plain phrase is unchanged", () => {
    expect(escapeLikePattern("aldi")).toBe("aldi");
  });

  test("LIKE wildcards in user input are escaped so they match literally", () => {
    expect(escapeLikePattern("50% off")).toBe("50\\% off");
    expect(escapeLikePattern("a_b")).toBe("a\\_b");
    expect(escapeLikePattern("back\\slash")).toBe("back\\\\slash");
  });
});

/**
 * A stand-in coordinator that records the SQL it is asked to run and replays
 * canned rows, so the search paths can be exercised without a real database.
 */
function fakeCoordinator(rows: { id: number; text: string }[]) {
  let queries: string[] = [];
  return {
    queries,
    coordinator: {
      query: async (sql: string) => {
        queries.push(sql);
        // The index build selects id and text; the phrase match selects ids.
        if (/AS text/.test(sql)) {
          return rows;
        }
        let lowered = sql.toLowerCase();
        let matches = rows.filter((row) => {
          // Extract each LIKE pattern and apply it as a substring test.
          let patterns = Array.from(lowered.matchAll(/like '%(.*?)%' escape/g)).map((m) => m[1]);
          return patterns.every((p) => row.text.toLowerCase().includes(p.replace(/\\(.)/g, "$1")));
        });
        if (/ORDER BY "id"/.test(sql)) {
          matches = matches.slice().sort((a, b) => a.id - b.id);
        }
        let limit = sql.match(/LIMIT (\d+)/);
        if (limit != null) {
          matches = matches.slice(0, Number(limit[1]));
        }
        return matches.map((row) => ({ id: row.id }));
      },
    } as any,
  };
}

// Mirrors the report in issue #137: the fuzzy encoder maps "aldi" and "aldea"
// to the same tokens, so a plain search for "aldi" surfaces "ALDEA HOMES"
// before the real "ALDI" rows.
const rows = [
  { id: 1, text: "ALDEA HOMES" },
  { id: 2, text: "ALDEA HOMES TWO" },
  { id: 3, text: "ALDI Supermarket" },
  { id: 4, text: "Corner ALDI" },
  { id: 5, text: "Walmart" },
  { id: 6, text: "ALDI store downtown" },
];

function searcher(rows: { id: number; text: string }[], fuzzyHits?: number[]) {
  let { coordinator, queries } = fakeCoordinator(rows);
  let s = new FullTextSearcher(coordinator, "points", { id: "id", text: "text" });
  if (fuzzyHits != null) {
    // The fuzzy index lives in a Worker, which is not available here, so stand
    // in for it with the ids flexsearch would have returned for the free text.
    (s as any).backendPromise = Promise.resolve({
      clear: async () => {},
      addPoints: async () => {},
      query: async () => fuzzyHits,
    });
  }
  return { queries, searcher: s };
}

describe("FullTextSearcher exact-phrase search", () => {
  test("a quoted query matches only the exact substring, case-insensitively", async () => {
    let { searcher: s } = searcher(rows);
    let result = await s.fullTextSearch('"aldi"', { limit: 100 });
    expect(new Set(result.map((r) => r.id))).toEqual(new Set([3, 4, 6]));
  });

  test("a phrase-only query never builds the fuzzy index", async () => {
    let { searcher: s, queries } = searcher(rows);
    await s.fullTextSearch('"aldi"', { limit: 100 });
    // The index build is the only query that selects the text column.
    expect(queries.some((q) => /AS text/.test(q))).toBe(false);
  });

  test("multiple phrases must all be present", async () => {
    let { searcher: s } = searcher(rows);
    expect((await s.fullTextSearch('"aldi" "store"', { limit: 100 })).map((r) => r.id)).toEqual([6]);
    expect(await s.fullTextSearch('"aldi" "walmart"', { limit: 100 })).toEqual([]);
  });

  test("a quoted query respects the limit and orders by id so repeats are stable", async () => {
    let { searcher: s, queries } = searcher(rows);
    let result = await s.fullTextSearch('"aldi"', { limit: 2 });
    expect(result.map((r) => r.id)).toEqual([3, 4]);
    expect(queries[0]).toMatch(/ORDER BY "id" LIMIT 2/);
  });

  test("a mixed query requires the phrase and fuzzy-matches the free text", async () => {
    // The fuzzy hits for "store" are rows 5 and 6, but only row 6 also contains
    // the exact phrase "aldi", so the phrase acts as a filter on the ranking.
    let { searcher: s } = searcher(rows, [5, 6]);
    let result = await s.fullTextSearch('"aldi" store', { limit: 100 });
    expect(new Set(result.map((r) => r.id))).toEqual(new Set([6]));
  });

  test("a mixed query preserves the fuzzy ranking order", async () => {
    let { searcher: s } = searcher(rows, [6, 4, 3, 1]);
    let result = await s.fullTextSearch('"aldi" store', { limit: 100 });
    // Rows 1 has no "aldi", the rest keep the order the fuzzy index gave.
    expect(result.map((r) => r.id)).toEqual([6, 4, 3]);
  });

  test("a mixed query applies the limit after the phrase filter", async () => {
    let { searcher: s } = searcher(rows, [1, 6, 4, 3]);
    let result = await s.fullTextSearch('"aldi" store', { limit: 2 });
    expect(result.map((r) => r.id)).toEqual([6, 4]);
  });

  test("a mixed query never serializes the candidate ids into the SQL", async () => {
    // The candidate set for a common word can be most of the table, so it is
    // intersected in JS rather than handed to the database as an IN list.
    let { searcher: s, queries } = searcher(rows, [6, 4, 3, 1]);
    await s.fullTextSearch('"aldi" store', { limit: 100 });
    let phraseQueries = queries.filter((q) => /LIKE/.test(q));
    expect(phraseQueries.length).toBe(1);
    expect(phraseQueries[0]).not.toMatch(/ IN \[/);
    expect(phraseQueries[0]).not.toMatch(/LIMIT/);
  });

  test("a mixed query with no fuzzy hits skips the phrase scan", async () => {
    let { searcher: s, queries } = searcher(rows, []);
    expect(await s.fullTextSearch('"aldi" store', { limit: 100 })).toEqual([]);
    expect(queries.some((q) => /LIKE/.test(q))).toBe(false);
  });

  test("wildcards in a phrase are matched literally", async () => {
    let { searcher: s } = searcher([
      { id: 1, text: "50% off today" },
      { id: 2, text: "50 percent off" },
    ]);
    let result = await s.fullTextSearch('"50%"', { limit: 100 });
    expect(result.map((r) => r.id)).toEqual([1]);
  });
});
