// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { resolveSQLTemplate, type ColumnDesc } from "../../utils/database.js";
import type { SQLField, SQLTable } from "../spec/spec.js";

export function fieldExpr(field: SQLField, context: { table: string }): SQL.ExprNode {
  let vars = { table: context.table, filter: "(true)" };
  if (typeof field == "string") {
    return SQL.column(field);
  } else {
    return SQL.sql`${resolveSQLTemplate(field.sql, vars)}`;
  }
}

export function fromExpr(table: SQLTable, context: { table: string; predicate?: string | null }): SQL.FromExpr {
  let vars = { table: context.table, filter: context.predicate ?? "(true)" };
  if (typeof table == "string") {
    return new SQL.TableRefNode(table);
  } else {
    return SQL.sql`(${resolveSQLTemplate(table.sql, vars)})`;
  }
}

/**
 * Whether a column can be used as the features column: either a string list, or a
 * list of structs with a VARCHAR `feature` field. Static check on the DB type string
 * (the runtime kind detection in the store uses DESCRIBE instead).
 */
export function isFeaturesColumn(column: ColumnDesc): boolean {
  if (column.jsType == "string[]") {
    return true;
  }
  return /^STRUCT\(.*\b"?feature"?\s+VARCHAR\b.*\)\[\]$/i.test(column.type);
}

/**
 * Columns with at most `maxClasses` distinct values, i.e., suitable for the predict field.
 * Predicting a high-cardinality column (e.g., raw text) yields one class per value, which
 * can freeze the page. Values are cast to TEXT as in predict mode.
 */
export async function lowCardinalityColumns(
  coordinator: Coordinator,
  table: string,
  columns: string[],
  maxClasses: number,
): Promise<string[]> {
  let count = async (aggregate: (value: SQL.ExprNode) => SQL.ExprNode, columns: string[]) => {
    if (columns.length == 0) {
      return [];
    }
    let select: Record<string, SQL.ExprNode> = {};
    columns.forEach((c, i) => {
      select[`c${i}`] = aggregate(SQL.sql`${SQL.column(c)}::TEXT`);
    });
    let row = ((await coordinator.query(SQL.Query.from(table).select(select))) as any).get(0);
    return columns.map((_, i) => Number(row[`c${i}`]));
  };
  // approx_count_distinct (HyperLogLog) can be off by a few even at small cardinalities, so use it
  // with slack to cheaply rule out high-cardinality columns, then count the rest exactly.
  let approx = await count((v) => SQL.sql`approx_count_distinct(${v})`, columns);
  let candidates = columns.filter((_, i) => approx[i] <= maxClasses * 2);
  let exact = await count((v) => SQL.sql`COUNT(DISTINCT ${v})`, candidates);
  return candidates.filter((_, i) => exact[i] <= maxClasses);
}
