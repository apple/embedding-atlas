// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import { literal, sql } from "@uwdata/mosaic-sql";

export type ExportFormat = "json" | "jsonl" | "csv" | "parquet";

const formats: Record<ExportFormat, { query: string; extension: string }> = {
  json: { query: "(FORMAT JSON, ARRAY true)", extension: "json" },
  jsonl: { query: "(FORMAT JSON)", extension: "jsonl" },
  csv: { query: "(FORMAT CSV)", extension: "csv" },
  parquet: { query: "(FORMAT parquet)", extension: "parquet" },
};

function makeQuery(table: string, predicate: string | null, name: string, format: ExportFormat) {
  if (predicate == null || predicate.length == 0) {
    return sql`COPY ${table} TO ${literal(name)} ${formats[format].query}`;
  } else {
    return sql`COPY (SELECT * FROM ${table} WHERE ${predicate}) TO ${literal(name)} ${formats[format].query}`;
  }
}

export function filenameForSelection(format: ExportFormat) {
  let name = `selection-${Date.now()}.${formats[format].extension}`;
  return name;
}

export async function exportMosaicSelection(
  coordinator: Coordinator,
  table: string,
  predicate: string | null,
  format: ExportFormat,
): Promise<[Uint8Array<ArrayBuffer>, string]> {
  let db = await (coordinator.databaseConnector() as any)?.getDuckDB();
  let name = filenameForSelection(format);
  await db.registerEmptyFileBuffer(name);
  let query = makeQuery(table, predicate, name, format);
  await coordinator.query(query.toString());
  let parquetBytes = await db.copyFileToBuffer(name);
  db.dropFile(name);
  return [parquetBytes, name];
}
