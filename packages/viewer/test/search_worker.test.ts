// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

/** Mirrors the worker-side filter so we can unit-test the false-positive case from #137. */
function textMatchesQuery(text: string, query: string): boolean {
  return text.toLowerCase().includes(query.toLowerCase());
}

function filterSearchResults(
  ids: (string | number)[],
  texts: Map<string | number, string>,
  query: string,
  limit: number,
): (string | number)[] {
  let normalizedQuery = query.trim();
  let matched: (string | number)[] = [];
  for (let id of ids) {
    let text = texts.get(id);
    if (text != null && textMatchesQuery(text, normalizedQuery)) {
      matched.push(id);
      if (matched.length >= limit) {
        break;
      }
    }
  }
  return matched;
}

describe("full-text search post-filter", () => {
  it("drops ALDEA when searching for aldi (#137)", () => {
    let texts = new Map<string | number, string>([
      [1, "ALDEA HOMES"],
      [2, "ALDI NORD"],
      [3, "ALDI SUD"],
    ]);
    // flexsearch forward indexing can rank ALDEA ahead of ALDI for query "aldi".
    let rankedIds = [1, 2, 3];
    expect(filterSearchResults(rankedIds, texts, "aldi", 10)).toEqual([2, 3]);
  });

  it("keeps case-insensitive substring matches", () => {
    let texts = new Map<string | number, string>([[1, "Visit ALDI today"]]);
    expect(filterSearchResults([1], texts, "aldi", 10)).toEqual([1]);
  });
});
