// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as duckdb from "@duckdb/duckdb-wasm/blocking";
import { createRequire } from "node:module";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, test } from "vitest";

import { FullTextSearcher } from "../src/search/search.js";

// The unit tests in search_index.test.ts replay the generated SQL against a
// regex stub, which checks the shape of the statement but not that DuckDB
// accepts it. These run the same statements against a real duckdb-wasm
// instance, the engine the viewer ships, to cover the two pieces a stub
// cannot: that `ESCAPE '\'` makes `%`, `_` and `\` in a phrase literal, and
// that the ids DuckDB hands back intersect with the ids the fuzzy index
// returns for the mixed path.

const require = createRequire(import.meta.url);
const dist = path.dirname(require.resolve("@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm"));

let db: duckdb.DuckDBBindings;
let conn: duckdb.DuckDBConnection;

const rows: [number, string][] = [
  [1, "ALDEA HOMES"],
  [2, "ALDI Supermarket"],
  [3, "Corner ALDI"],
  [4, "ALDI store downtown"],
  [5, "50% off today"],
  [6, "50 percent off"],
  [7, "a_b"],
  [8, "axb"],
  [9, "back\\slash"],
  [10, "backslash"],
];

beforeAll(async () => {
  db = await duckdb.createDuckDB(
    {
      mvp: { mainModule: `${dist}/duckdb-mvp.wasm`, mainWorker: `${dist}/duckdb-node-mvp.worker.cjs` },
      eh: { mainModule: `${dist}/duckdb-eh.wasm`, mainWorker: `${dist}/duckdb-node-eh.worker.cjs` },
    },
    new duckdb.VoidLogger(),
    duckdb.NODE_RUNTIME,
  );
  await db.instantiate(() => {});
  conn = db.connect();
  conn.query(`CREATE TABLE points (id BIGINT, text VARCHAR)`);
  let values = rows.map(([id, text]) => `(${id}, '${text.replace(/'/g, "''")}')`).join(", ");
  conn.query(`INSERT INTO points VALUES ${values}`);
}, 30_000);

afterAll(() => {
  conn?.close();
});

// The searcher only needs `query`, which returns an iterable of rows.
function coordinator() {
  let queries: string[] = [];
  return {
    queries,
    coordinator: { query: async (sql: string) => (queries.push(sql), conn.query(sql)) } as any,
  };
}

function ids(result: { id: any }[]): number[] {
  return result.map((r) => Number(r.id));
}

describe("FullTextSearcher against duckdb-wasm", () => {
  test("a phrase is matched as a case-insensitive substring", async () => {
    let s = new FullTextSearcher(coordinator().coordinator, "points", { id: "id", text: "text" });
    expect(ids(await s.fullTextSearch('"aldi"', { limit: 100 }))).toEqual([2, 3, 4]);
    expect(ids(await s.fullTextSearch('"aldi" "store"', { limit: 100 }))).toEqual([4]);
  });

  test("LIKE wildcards and the escape character in a phrase are literal", async () => {
    let s = new FullTextSearcher(coordinator().coordinator, "points", { id: "id", text: "text" });
    expect(ids(await s.fullTextSearch('"50%"', { limit: 100 }))).toEqual([5]);
    expect(ids(await s.fullTextSearch('"a_b"', { limit: 100 }))).toEqual([7]);
    expect(ids(await s.fullTextSearch('"back\\slash"', { limit: 100 }))).toEqual([9]);
  });

  test("the limit is applied in id order so a repeated search is stable", async () => {
    let s = new FullTextSearcher(coordinator().coordinator, "points", { id: "id", text: "text" });
    let first = ids(await s.fullTextSearch('"aldi"', { limit: 2 }));
    let second = ids(await s.fullTextSearch('"aldi"', { limit: 2 }));
    expect(first).toEqual([2, 3]);
    expect(second).toEqual(first);
  });

  test("a predicate narrows the phrase match", async () => {
    let s = new FullTextSearcher(coordinator().coordinator, "points", { id: "id", text: "text" });
    expect(ids(await s.fullTextSearch('"aldi"', { limit: 100, predicate: "id > 2" }))).toEqual([3, 4]);
  });

  test("a mixed query intersects the fuzzy ids with the ids DuckDB returns", async () => {
    let { coordinator: c, queries } = coordinator();
    let s = new FullTextSearcher(c, "points", { id: "id", text: "text" });
    // Stand in for the fuzzy index with ids of the same type DuckDB produces
    // for a BIGINT column, in the order flexsearch would have ranked them.
    let fuzzy = Array.from(conn.query(`SELECT id FROM points WHERE id IN [4, 1, 3, 6]`)).map((r: any) => r.id);
    expect(fuzzy.every((id) => typeof id == "bigint")).toBe(true);
    (s as any).backendPromise = Promise.resolve({
      clear: async () => {},
      addPoints: async () => {},
      query: async () => fuzzy,
    });
    let result = await s.fullTextSearch('"aldi" store', { limit: 100 });
    expect(result.map((r) => r.id)).toEqual(fuzzy.filter((id) => [3n, 4n].includes(id)));
    expect(queries.filter((q) => /LIKE/.test(q)).every((q) => !/ IN \[/.test(q))).toBe(true);
  });
});
