// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";

import { isWideIntegerDBType } from "../../utils/db_types.js";

/**
 * Cast wide integers only after sorting and pagination. Flechette otherwise
 * coerces Arrow INT64/UINT64 values to numbers and rejects values outside
 * JavaScript's safe integer range.
 */
export function instancesDisplayQuery(
  query: SQL.SelectQuery,
  columns: string[],
  columnTypes: ReadonlyMap<string, string>,
): SQL.SelectQuery {
  if (!columns.some((column) => isWideIntegerDBType(columnTypes.get(column) ?? ""))) {
    return query;
  }

  return SQL.Query.from(query).select(
    Object.fromEntries(
      columns.map((column) => [
        column,
        isWideIntegerDBType(columnTypes.get(column) ?? "")
          ? SQL.cast(SQL.column(column), "VARCHAR")
          : SQL.column(column),
      ]),
    ),
  );
}
