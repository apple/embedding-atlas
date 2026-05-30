// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Coordinator, restConnector, socketConnector, wasmConnector, type Selection } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { isFloatingPointDBType, jsTypeFromDBType, type ColumnDesc, type JSType } from "./db_types.js";
import { createDuckDB } from "./duckdb.js";

// Re-export the pure type-classification helpers so existing imports from
// "./database.js" keep working.
export { isFloatingPointDBType, jsTypeFromDBType, type ColumnDesc, type JSType };

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

export function resolveSQLTemplate(template: string, vars: Record<string, string>): string {
  return template.replace(/\$([a-zA-Z][a-zA-Z0-9\_]+)/g, (original, name) => {
    if (vars[name] != undefined) {
      return vars[name];
    } else {
      return original;
    }
  });
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
