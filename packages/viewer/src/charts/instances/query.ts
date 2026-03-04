// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";

import { predicateToString, resolveSQLTemplate } from "../../utils/database.js";

// Helper to build a query with automatic predicate handling
// For custom queries: $predicate is substituted into the query string
// For normal tables: predicate is applied via .where()
export function instancesQuery(options: {
  query?: string;
  table: string;
  predicate?: SQL.FilterExpr | null;
}): SQL.SelectQuery {
  if (options.query) {
    // Custom query: substitute $table and $predicate in the query string
    const predicateStr = options.predicate ? predicateToString(options.predicate) : null;
    const queryStr = resolveSQLTemplate(options.query, { table: options.table, filter: predicateStr ?? "(true)" });
    const from = SQL.sql`(${queryStr})`;
    return from as any;
  } else {
    let q = SQL.Query.from(options.table).select("*");
    if (options.predicate) {
      q = q.where(options.predicate);
    }
    return q;
  }
}
