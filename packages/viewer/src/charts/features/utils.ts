import * as SQL from "@uwdata/mosaic-sql";

import { resolveSQLTemplate } from "../../utils/database.js";
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
