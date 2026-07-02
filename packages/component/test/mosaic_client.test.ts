// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";
import { describe, expect, it } from "vitest";

import {
  DataPointQuery,
  predicateForDataPoints,
  queryApproximateDensity,
} from "../src/lib/embedding_view/mosaic_client.js";
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

  // A coordinator stub that records the SQL AND returns a fixed row, so the row-found
  // branches (conversion, rowid attachment) are exercised.
  function fakeCoordinatorReturning(row: any) {
    let captured: string[] = [];
    return {
      captured,
      query: async (q: any) => {
        captured.push(String(q));
        return { get: () => row };
      },
    };
  }

  // A coordinator stub for the 3D coordinate-fallback path: it answers the ambiguity
  // COUNT(DISTINCT identifier) query with `distinctIds` and the row query with `row`.
  function fakeCoordinator3DPick(distinctIds: number, row: any) {
    let captured: string[] = [];
    return {
      captured,
      query: async (q: any) => {
        let sql = String(q);
        captured.push(sql);
        if (sql.includes("COUNT(DISTINCT")) {
          return { get: () => ({ n: distinctIds }) };
        }
        return { get: () => row };
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

  it("queryByRowId resolves the row and attaches the rowid as a JS number", async () => {
    let coord = fakeCoordinatorReturning({ x: 1, y: 2, z: 3, category: 0 });
    let q = new DataPointQuery(coord as any, source);
    let point = await q.queryByRowId(7n);
    expect(point).not.toBeNull();
    expect(point!.rowid).toBe(7);
    expect(point!.x).toBe(1);
  });

  it("queryByRowId preserves a large rowid exactly through the Number() conversion", async () => {
    let coord = fakeCoordinatorReturning({ x: 0, y: 0 });
    let q = new DataPointQuery(coord as any, source);
    let point = await q.queryByRowId(9007199254740991n); // 2^53 - 1
    expect(point!.rowid).toBe(9007199254740991);
  });

  it("queryByRowId AND-composes the active cross-filter predicate (anti-resurrection)", async () => {
    let coord = fakeCoordinatorReturning({ x: 0, y: 0 });
    let q = new DataPointQuery(coord as any, source);
    await q.queryByRowId(7n, SQL.sql`"flag" = 1`);
    let sql = coord.captured.join("\n");
    expect(sql).toContain("rowid = 7");
    expect(sql).toContain('"flag" = 1');
  });

  it("queryClosestPoint3D brackets all three axes with a FLOAT cast and orders by 3D distance", async () => {
    let coord = fakeCoordinator(); // returns null so both radii are attempted
    let q = new DataPointQuery(coord as any, source);
    await q.queryClosestPoint3D(null, 1.5, 2.5, 3.5, 0.01);
    let sql = coord.captured.join("\n");
    expect(sql).toContain('CAST("xc" AS FLOAT) BETWEEN');
    expect(sql).toContain('CAST("yc" AS FLOAT) BETWEEN');
    expect(sql).toContain('CAST("zc" AS FLOAT) BETWEEN');
    expect(sql).toContain("z - (3.5)"); // orderby includes the z term
  });

  it("queryClosestPoint3D delegates to the 2D closest-point query when the source has no z", async () => {
    let coord = fakeCoordinator();
    let q = new DataPointQuery(coord as any, { ...source, z: null });
    await q.queryClosestPoint3D(null, 1, 2, 3, 0.01);
    let sql = coord.captured.join("\n");
    expect(sql).not.toContain("zc"); // no z column referenced at all
    expect(sql).not.toContain("AS FLOAT"); // 2D path brackets x/y without the FLOAT cast
  });

  it("queryClosestPoint3D AND-composes the active cross-filter predicate", async () => {
    let coord = fakeCoordinator();
    let q = new DataPointQuery(coord as any, source);
    await q.queryClosestPoint3D(SQL.sql`"flag" = 1`, 1, 2, 3, 0.01);
    let sql = coord.captured.join("\n");
    expect(sql).toContain('"flag" = 1');
  });

  it("queryClosestPoint3D returns the exact-match row when it is unique", async () => {
    let coord = fakeCoordinator3DPick(1, { x: 1, y: 2, z: 3, identifier: "a" });
    let q = new DataPointQuery(coord as any, source);
    let result = await q.queryClosestPoint3D(null, 1, 2, 3, 0.01);
    expect(result).not.toBeNull();
    expect(result!.identifier).toBe("a");
  });

  it("queryClosestPoint3D resolves to indeterminate when distinct records collide (COUNT DISTINCT covers 3+ rows)", async () => {
    // Distinct rows share the rendered x/y/z; picking one would target the wrong
    // identifier, so the pick must be indeterminate (null).
    let coord = fakeCoordinator3DPick(2, { x: 1, y: 2, z: 3, identifier: "a" });
    let q = new DataPointQuery(coord as any, source);
    let result = await q.queryClosestPoint3D(null, 1, 2, 3, 0.01);
    expect(result).toBeNull();
    // Ambiguity is decided by a DISTINCT-identifier count over ALL colliding rows, not a
    // LIMIT-2 peek that could miss a distinct sibling behind same-identifier duplicates.
    expect(coord.captured.join("\n")).toContain('COUNT(DISTINCT "idc")');
  });

  it("queryClosestPoint3D still resolves when co-located rows share the same identifier", async () => {
    // One distinct identifier => unambiguous (the coordinated predicate targets the same
    // record either way); resolve normally instead of returning indeterminate.
    let coord = fakeCoordinator3DPick(1, { x: 1, y: 2, z: 3, identifier: "a" });
    let q = new DataPointQuery(coord as any, source);
    let result = await q.queryClosestPoint3D(null, 1, 2, 3, 0.01);
    expect(result).not.toBeNull();
    expect(result!.identifier).toBe("a");
  });
});

describe("queryApproximateDensity", () => {
  it("handles DuckDB BIGINT count aggregates (JS bigint) without throwing", async () => {
    // SUM(count)/MAX(count) are cast to BIGINT to avoid INT32 overflow, so DuckDB returns
    // them as JS bigint. The density/count math must convert them to number rather than
    // throw on bigint-vs-number arithmetic (this ran before any embedding view renders).
    let row = { centerX: 1, centerY: 2, stdX: 3, stdY: 4, maxCategory: 2, maxCount: 5n, totalCount: 100n };
    let coord = { query: async () => ({ get: () => row }) };
    let result = await queryApproximateDensity(coord as any, { table: "t", x: "xc", y: "yc", category: "cc" });
    expect(typeof result.totalCount).toBe("number");
    expect(result.totalCount).toBe(100);
    expect(typeof result.maxDensity).toBe("number");
    expect(Number.isFinite(result.maxDensity)).toBe(true);
    expect(result.categoryCount).toBe(3);
  });
});
