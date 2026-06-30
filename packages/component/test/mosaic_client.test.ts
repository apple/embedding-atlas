// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it } from "vitest";

import { DataPointQuery, predicateForDataPoints } from "../src/lib/embedding_view/mosaic_client.js";
import type { DataPoint } from "../src/lib/embedding_view/types.js";

describe("predicateForDataPoints", () => {
  let source = { x: "xc", y: "yc", z: "zc", category: "cc", identifier: null as string | null };

  it("targets the exact row by rowid for a co-located 3D pick without an identifier", () => {
    // Two rows share x/y/z/category and there is no identifier column. Selecting the
    // one picked row (resolved by rowid) must produce a predicate matching ONLY that
    // row — not the coordinate equality that would also match its stacked twin.
    let picked: DataPoint = { x: 1.5, y: 2.5, z: 3.5, category: 0, rowid: 7 };
    let sql = String(predicateForDataPoints(source, [picked]));
    expect(sql).toContain("rowid = 7");
    expect(sql).not.toContain("::DOUBLE"); // did not fall back to x/y/z/category equality
  });

  it("ORs distinct rowids when several rowid-resolved points are selected", () => {
    let points: DataPoint[] = [
      { x: 1, y: 1, z: 1, category: 0, rowid: 7 },
      { x: 1, y: 1, z: 1, category: 0, rowid: 12 },
    ];
    let sql = String(predicateForDataPoints(source, points));
    expect(sql).toContain("rowid = 7");
    expect(sql).toContain("rowid = 12");
  });

  it("falls back to coordinate equality when no point carries a rowid", () => {
    let points: DataPoint[] = [{ x: 1.5, y: 2.5, z: 3.5, category: 0 }];
    let sql = String(predicateForDataPoints(source, points));
    expect(sql).not.toContain("rowid");
    expect(sql).toContain("::DOUBLE");
  });

  it("does not use rowid when only SOME points carry one (avoids a partial-identity predicate)", () => {
    let points: DataPoint[] = [
      { x: 1, y: 1, z: 1, category: 0, rowid: 7 },
      { x: 2, y: 2, z: 2, category: 0 }, // no rowid
    ];
    let sql = String(predicateForDataPoints(source, points));
    expect(sql).not.toContain("rowid =");
    expect(sql).toContain("::DOUBLE");
  });

  it("prefers an identifier over coordinates when rowid is absent", () => {
    let withId = { ...source, identifier: "id" };
    let points: DataPoint[] = [{ x: 1, y: 1, z: 1, category: 0, identifier: "abc" }];
    let sql = String(predicateForDataPoints(withId, points));
    expect(sql).toContain("id");
    expect(sql).toContain("abc");
    expect(sql).not.toContain("::DOUBLE");
  });

  it("prefers the semantic identifier over rowid when both are present", () => {
    // A configured identifier is stable across views/joins/reloads/other clients;
    // the physical, table-local rowid must NOT leak into a coordinated predicate.
    let withId = { ...source, identifier: "id" };
    let points: DataPoint[] = [{ x: 1, y: 1, z: 1, category: 0, identifier: "abc", rowid: 7 }];
    let sql = String(predicateForDataPoints(withId, points));
    expect(sql).toContain("abc");
    expect(sql).not.toContain("rowid");
    expect(sql).not.toContain("::DOUBLE");
  });

  it("returns a constant-false predicate for an empty selection", () => {
    let sql = String(predicateForDataPoints(source, []));
    expect(sql.toUpperCase()).toContain("FALSE");
  });
});

describe("DataPointQuery exact-identity resolution", () => {
  // A coordinator stub that records the SQL it is handed (so we can assert HOW a pick
  // is resolved) and returns no rows.
  function fakeCoordinator() {
    let captured: string[] = [];
    return {
      captured,
      query: async (q: any) => {
        captured.push(String(q));
        return { get: () => null };
      },
    };
  }

  let source = { table: "t", x: "xc", y: "yc", z: "zc", category: "cc", identifier: "idc" };

  it("queryByRowId resolves strictly by the physical rowid", async () => {
    let coord = fakeCoordinator();
    let q = new DataPointQuery(coord as any, source);
    await q.queryByRowId(7n);
    let sql = coord.captured.join("\n");
    expect(sql).toContain("rowid = 7");
  });

  it("queryByIdentifier resolves a rowid-less pick by the identifier column (not coordinates)", async () => {
    // The fallback for views/joins: an exact identifier lookup, so a pick on one of
    // several co-located rows targets the precise row instead of an arbitrary twin.
    let coord = fakeCoordinator();
    let q = new DataPointQuery(coord as any, source);
    await q.queryByIdentifier("abc");
    let sql = coord.captured.join("\n");
    expect(sql).toContain('"idc" = ');
    expect(sql).toContain("abc");
    expect(sql).not.toContain("rowid");
    expect(sql).not.toContain("BETWEEN"); // not a coordinate-range lookup
  });

  it("queryByIdentifier no-ops (no query) when the source has no identifier column", async () => {
    let coord = fakeCoordinator();
    let q = new DataPointQuery(coord as any, { ...source, identifier: null });
    let result = await q.queryByIdentifier("abc");
    expect(result).toBeNull();
    expect(coord.captured.length).toBe(0);
  });
});
