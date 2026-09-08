// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { streamingRestConnector } from "@embedding-atlas/component";
import { Coordinator, socketConnector, wasmConnector, type Selection } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { isFloatingPointDBType, jsTypeFromDBType, type ColumnDesc, type JSType, type TableDesc } from "./db_types.js";
import { createDuckDB } from "./duckdb.js";

// Re-export the pure type-classification helpers so existing imports from
// "./database.js" keep working.
export { isFloatingPointDBType, jsTypeFromDBType, type ColumnDesc, type JSType, type TableDesc };

/** Initialize the database connector for a Mosaic coordinator.
 *
 *  The "rest" path uses ``streamingRestConnector`` from
 *  ``@embedding-atlas/component`` instead of Mosaic's stock
 *  ``restConnector``. The stock version calls ``Response.arrayBuffer()``
 *  on the arrow response, which Chrome aborts past ~2 GB — at the
 *  322 M-row eubucco file the (zstd-compressed u32) scatter response
 *  is ~2.1 GB and was getting us a "Failed to fetch" with no points
 *  on the canvas. The streaming variant reads via ``response.body``
 *  into a manually-allocated ``Uint8Array``, which goes through V8's
 *  larger ~4 GB ArrayBuffer cap. */
export async function initializeDatabase(
  coordinator: Coordinator,
  type: "wasm" | "socket" | "rest",
  uri: string | null | undefined = undefined,
) {
  const db = await createDuckDB();
  if (type == "wasm") {
    const conn = await wasmConnector({ duckdb: db.duckdb, connection: db.connection });
    coordinator.databaseConnector(conn);
  } else if (type == "socket") {
    const conn = await socketConnector({ uri: uri ?? "" });
    coordinator.databaseConnector(conn);
  } else if (type == "rest") {
    const conn = streamingRestConnector({ uri: uri ?? "" });
    coordinator.databaseConnector(conn);
  }
}

/** Convert a Mosaic predicate to SQL string */
export function predicateToString(predicate: ReturnType<Selection["predicate"]>): string | null {
  if (predicate == null) {
    return null;
  }
  if (predicate instanceof Array) {
    if (predicate.length == 0) {
      return null;
    }
    return SQL.and(predicate).toString().trim();
  }
  if (typeof predicate == "string") {
    return predicate.trim();
  }
  if (typeof predicate == "boolean") {
    return SQL.literal(predicate).toString();
  }
  return predicate.toString().trim();
}

export function resolveSQLTemplate(template: string, vars: Record<string, string>): string {
  return template.replace(/\$([a-zA-Z][a-zA-Z0-9\_]+)/g, (original, name) => {
    if (vars[name] != undefined) {
      return vars[name];
    } else {
      return original;
    }
  });
}

/** Format a DuckDB column type for compact display in dropdown labels.
 *  ENUMs are abbreviated to ``ENUM`` instead of dumping the full value list,
 *  which can be hundreds of strings for high-cardinality categorical cols. */
export function formatColumnType(t: string): string {
  if (t.startsWith("ENUM(")) return "ENUM";
  return t;
}

export interface EmbeddingLegend {
  indexColumn: string;
  legend: {
    label: string;
    color: string;
    predicate: any;
    count: number;
  }[];
}

export async function columnDescriptions(coordinator: Coordinator, table: string): Promise<ColumnDesc[]> {
  let result = Array.from(await coordinator.query(`DESCRIBE ${table}`));
  return result.map((column) => ({
    name: column.column_name,
    type: column.column_type,
    jsType: jsTypeFromDBType(column.column_type),
  }));
}

export async function distinctCount(coordinator: Coordinator, table: string, column: string): Promise<number> {
  // APPROX_COUNT_DISTINCT (HyperLogLog) is ~10× faster than exact
  // ``COUNT(DISTINCT col)`` on multi-million-row datasets and accurate to
  // ~1 % — the threshold checks that consume this value (skip-if-≤1,
  // count-plot-if-≤1000, count-plot-if-≤10) all tolerate that error.
  let r = await coordinator.query(`SELECT APPROX_COUNT_DISTINCT(${SQL.column(column)}) AS count FROM ${table}`);
  return r.get(0).count;
}
/** Batch ``distinctCount`` for many columns into a single fused-aggregate
 *  scan. On a 75 M-row table this is ~30× faster than calling
 *  ``distinctCount`` in a loop (single parquet pass vs. N passes plus
 *  N round-trips), e.g. 11 cols: 5.7 s sequential → ~200 ms fused. */
export async function distinctCountBatch(
  coordinator: Coordinator,
  table: string,
  columns: string[],
): Promise<Map<string, number>> {
  let result = new Map<string, number>();
  if (columns.length === 0) return result;
  let sel = columns.map((c, i) => `APPROX_COUNT_DISTINCT(${SQL.column(c)}) AS "c${i}"`).join(", ");
  let row = (await coordinator.query(`SELECT ${sel} FROM ${table}`)).get(0);
  for (let i = 0; i < columns.length; i++) {
    let v = row[`c${i}`];
    result.set(columns[i], typeof v === "bigint" ? Number(v) : v);
  }
  return result;
}
