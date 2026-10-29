// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Selection } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";
import { describe, expect, it } from "vitest";
import { applySearchResultsSelection } from "../src/search/selection.js";

describe("search result selection", () => {
  it("adds result IDs alongside existing cross-filters", () => {
    const filter = Selection.crossfilter();
    const existingSource = {};
    const existingPredicate = SQL.eq(SQL.column("category"), SQL.literal("birds"));
    filter.update({
      source: existingSource,
      clients: new Set(),
      value: "birds",
      predicate: existingPredicate,
    });

    applySearchResultsSelection(filter, {}, "items", "id", [1, 2, 2]);

    expect(filter.clauses).toHaveLength(2);
    expect(filter.clauses[0].source).toBe(existingSource);
    expect(filter.clauses[0].predicate).toBe(existingPredicate);
    expect(filter.clauses[1].value).toEqual([1, 2]);
    expect(filter.clauses[1].clients?.size).toBe(0);
    expect(filter.predicate({} as any)).toHaveLength(2);
  });

  it("replaces only its own previous result clause", () => {
    const filter = Selection.crossfilter();
    const source = {};

    applySearchResultsSelection(filter, source, "items", "id", [1, 2]);
    applySearchResultsSelection(filter, source, "items", "id", [3]);

    expect(filter.clauses).toHaveLength(1);
    expect(filter.clauses[0].source).toBe(source);
    expect(filter.clauses[0].value).toEqual([3]);
  });

  it("leaves the current selection unchanged for an empty result set", () => {
    const filter = Selection.crossfilter();
    const existingSource = {};
    filter.update({ source: existingSource, clients: new Set(), value: "keep", predicate: SQL.literal(true) });

    applySearchResultsSelection(filter, {}, "items", "id", []);

    expect(filter.clauses).toHaveLength(1);
    expect(filter.clauses[0].source).toBe(existingSource);
  });
});
