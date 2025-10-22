// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Coordinator, restConnector, socketConnector, wasmConnector, type Selection } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { createDuckDB } from "./duckdb.js";

/** Initialize the database connector for a Mosaic coordinator */
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
    const conn = await restConnector({ uri: uri ?? "" });
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

/** Column description */
export interface ColumnDesc {
  name: string;
  type: string;
  jsType: JSType | null;
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
  let r = await coordinator.query(`SELECT COUNT(DISTINCT ${SQL.column(column)}) AS count FROM ${table}`);
  return r.get(0).count;
}

export type JSType = "string" | "number" | "string[]";

export function jsTypeFromDBType(dbType: string): JSType | null {
  if (numberTypes.has(dbType)) {
    return "number";
  } else if (stringTypes.has(dbType)) {
    return "string";
  } else if (dbType.match(/^(VARCHAR|TEXT)\[\d*\]$/)) {
    return "string[]";
  } else {
    return null;
  }
}

const numberTypes = new Set([
  "REAL",
  "FLOAT4",
  "FLOAT8",
  "FLOAT",
  "DOUBLE",
  "INT",
  "TINYINT",
  "INT1",
  "SMALLINT",
  "INT2",
  "SHORT",
  "INTEGER",
  "INT4",
  "INT",
  "SIGNED",
  "INT8",
  "LONG",
  "BIGINT",
  "UTINYINT",
  "USMALLINT",
  "UINTEGER",
  "UBIGINT",
  "UHUGEINT",
]);

const stringTypes = new Set(["BOOLEAN", "DATE", "VARCHAR", "CHAR", "BPCHAR", "TEXT", "STRING"]);
