// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";

import { defaultCategoryColors } from "../../colors.js";

/** A column name, or a SQL expression (evaluated in the context of the layer's table). */
export type SQLField = string | { sql: string };

/** The SQL expression of a field. */
export function fieldExpr(field: SQLField): SQL.ExprNode {
  return typeof field == "string" ? SQL.column(field) : SQL.sql`(${field.sql})`;
}

/** A stable key of a field, for comparing fields and for cache keys. */
export function fieldKey(field: SQLField | null | undefined): string | null {
  if (field == null) {
    return null;
  }
  return typeof field == "string" ? `column:${field}` : `sql:${field.sql}`;
}

/** How the values of a category field map to categories (and colors).
 *
 *  - With `domain`: the values (compared as text) are the categories, in order.
 *  - With `thresholds` `[t0, t1, ..., tn]`: numeric (or time) values are binned into `[t0, t1), ..., [tn-1, tn]`.
 *  - Otherwise, the field's values are category indices (0, 1, ...), as with `categoryColors`.
 *
 *  With `domain` or `thresholds`, values outside them (including infinite values) are "other", and null (and NaN)
 *  values are "null"; their colors are `other` and `null`. */
export interface ColorScale {
  /** The category values, in order (compared as text). */
  domain?: (string | number | boolean)[] | null;
  /** Bin boundaries for numeric values, in increasing order (`n + 1` boundaries for `n` bins). */
  thresholds?: number[] | null;
  /** The type of values for `thresholds`: numbers (default), or times (compared as epoch milliseconds). */
  type?: "number" | "time" | null;
  /** The colors of the categories (or bins). Missing colors use the default category colors. */
  range?: string[] | null;
  /** The color of values outside the domain (or thresholds). */
  other?: string | null;
  /** The color of null (and NaN) values. */
  null?: string | null;
}

export const DEFAULT_OTHER_COLOR = "#9eabc2";
export const DEFAULT_NULL_COLOR = "#aaaaaa";

/** The maximum number of categories of a domain (or bins), so that "other" and "null" fit in a byte. */
export const MAX_SCALE_CATEGORIES = 254;

/** The number of categories (or bins) of a scale, or null if its values are category indices. */
export function scaleCategoryCount(scale: ColorScale | null | undefined): number | null {
  if (scale?.domain != null) {
    return Math.min(scale.domain.length, MAX_SCALE_CATEGORIES);
  }
  if (scale?.thresholds != null) {
    return Math.min(Math.max(scale.thresholds.length - 1, 0), MAX_SCALE_CATEGORIES);
  }
  return null;
}

/** The category index of a value, as a SQL expression: `0 ... k-1` for the categories (or bins), `k` for "other",
 *  `k + 1` for "null" (k = the number of categories). Values are category indices if the scale has neither a domain
 *  nor thresholds. The view and its legends (e.g., in the viewer) use this, so their categories always match. */
export function categoryIndexExpr(value: SQL.ExprNode, scale: ColorScale | null | undefined): SQL.ExprNode {
  let k = scaleCategoryCount(scale);
  if (k == null) {
    return value;
  }
  let other = SQL.literal(k);
  let nullIndex = SQL.literal(k + 1);
  if (scale!.domain != null) {
    let text = SQL.cast(value, "TEXT");
    let cases = scale!.domain
      .slice(0, k)
      .map((v, i) => SQL.sql`WHEN ${text} = ${SQL.literal(String(v))} THEN ${SQL.literal(i)}`);
    return SQL.sql`(CASE WHEN ${value} IS NULL THEN ${nullIndex} ${joinSQL(cases)} ELSE ${other} END)`;
  }
  let t = scale!.thresholds!.slice(0, k + 1);
  let x = scale!.type == "time" ? SQL.cast(SQL.epoch_ms(value), "DOUBLE") : SQL.cast(value, "DOUBLE");
  if (t.length < 2) {
    return SQL.sql`(CASE WHEN ${x} IS NULL OR isnan(${x}) THEN ${nullIndex} ELSE ${other} END)`;
  }
  let bins = t.slice(1, -1).map((boundary, i) => SQL.sql`WHEN ${x} < ${SQL.literal(boundary)} THEN ${SQL.literal(i)}`);
  return SQL.sql`(CASE WHEN ${x} IS NULL OR isnan(${x}) THEN ${nullIndex} WHEN ${x} < ${SQL.literal(t[0])} OR ${x} > ${SQL.literal(t[t.length - 1])} THEN ${other} ${joinSQL(bins)} ELSE ${SQL.literal(k - 1)} END)`;
}

/** SQL fragments joined with spaces. */
function joinSQL(parts: SQL.ExprNode[]): SQL.ExprNode | string {
  return parts.length == 0 ? "" : parts.reduce((a, b) => SQL.sql`${a} ${b}`);
}

/** The colors of a scale's category indices (see `categoryIndexExpr`): the categories' colors, then "other" and
 *  "null". Null if the scale's values are category indices (then `range`, or other colors, apply directly). */
export function scaleColors(scale: ColorScale | null | undefined): string[] | null {
  let k = scaleCategoryCount(scale);
  if (k == null) {
    return null;
  }
  let defaults = defaultCategoryColors(Math.max(k, 1));
  let colors: string[] = [];
  for (let i = 0; i < k; i++) {
    colors.push(scale!.range?.[i] ?? defaults[i]);
  }
  colors.push(scale!.other ?? DEFAULT_OTHER_COLOR);
  colors.push(scale!.null ?? DEFAULT_NULL_COLOR);
  return colors;
}
