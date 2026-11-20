// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as duckdb from "@duckdb/duckdb-wasm/blocking";
import { createRequire } from "node:module";
import { afterAll, beforeAll, describe, expect, test, vi } from "vitest";

import { importDataTable } from "../src/app/import_data.js";

const require = createRequire(import.meta.url);

let db: duckdb.DuckDBBindings;

beforeAll(async () => {
  // resolveAppConfig() reads window.EMBEDDING_ATLAS_CONFIG.
  vi.stubGlobal("window", {});
  db = await duckdb.createDuckDB(
    {
      mvp: {
        mainModule: require.resolve("@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm"),
        mainWorker: require.resolve("@duckdb/duckdb-wasm/dist/duckdb-node-mvp.worker.cjs"),
      },
      eh: {
        mainModule: require.resolve("@duckdb/duckdb-wasm/dist/duckdb-eh.wasm"),
        mainWorker: require.resolve("@duckdb/duckdb-wasm/dist/duckdb-node-eh.worker.cjs"),
      },
    },
    new duckdb.VoidLogger(),
    duckdb.NODE_RUNTIME,
  );
  await db.instantiate(() => {});
}, 30_000);

afterAll(() => {
  vi.unstubAllGlobals();
});

async function importFiles(files: File[]) {
  let conn = db.connect();
  conn.query("DROP TABLE IF EXISTS dataset");
  // importDataTable expects the async API; wrap the blocking bindings.
  let asyncDB = {
    registerFileBuffer: async (name: string, buffer: Uint8Array) => db.registerFileBuffer(name, buffer),
    dropFile: async (name: string) => db.dropFile(name),
  };
  let asyncConnection = { query: async (sql: string) => conn.query(sql) };
  await importDataTable(files, asyncDB as any, asyncConnection as any, "dataset");
  let result = conn.query("SELECT * FROM dataset ORDER BY ALL");
  let rows = result.toArray().map((row) => row.toJSON());
  let columns = result.schema.fields.map((f) => f.name);
  conn.close();
  return { columns, rows };
}

describe("importDataTable", () => {
  test("matches columns by name when files list them in a different order", async () => {
    let { columns, rows } = await importFiles([
      new File(["x,y,text\n1.5,10.5,a\n2.5,20.5,b\n"], "a.csv"),
      new File(["y,x,text\n30.5,3.5,c\n40.5,4.5,d\n"], "b.csv"),
    ]);
    expect(columns).toEqual(["x", "y", "text", "filename"]);
    expect(rows).toEqual([
      { x: 1.5, y: 10.5, text: "a", filename: "a.csv" },
      { x: 2.5, y: 20.5, text: "b", filename: "a.csv" },
      { x: 3.5, y: 30.5, text: "c", filename: "b.csv" },
      { x: 4.5, y: 40.5, text: "d", filename: "b.csv" },
    ]);
  });

  test("keeps a filename column that is already in the files", async () => {
    let { columns, rows } = await importFiles([
      new File(["filename,x\nq,1.5\n"], "a.csv"),
      new File(["filename,x\nr,2.5\n"], "b.csv"),
    ]);
    expect(columns).toEqual(["filename", "x", "filename_1"]);
    expect(rows).toEqual([
      { filename: "q", x: 1.5, filename_1: "a.csv" },
      { filename: "r", x: 2.5, filename_1: "b.csv" },
    ]);
  });
});
