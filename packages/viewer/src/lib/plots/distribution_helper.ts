// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { inferBinning } from "../charts/binning.js";
import type { ScaleSpec, ScaleType } from "../charts/types.js";
import { jsTypeFromDBType } from "../database_utils.js";

export interface DistributionStats {
  table: string;
  field: string;
  /** Available if the data is quantitative */
  quantitative?: {
    /** Number of finite values */
    count: number;
    /** The minimum finite value */
    min: number;
    /** The maximum finite value */
    max: number;
    /** The mean of finite values */
    mean: number;
    /** The median finite values */
    median: number;
    /** The minimum positive finite value */
    minPositive: number;

    /** Number of non-finite values (inf, nan, null) */
    countNonFinite: number;
  };
  /** Available if the data is nominal */
  nominal?: {
    // Top k levels
    levels: { value: string; count: number }[];
    // Number of other levels
    numOtherLevels: number;
    // Number of points in "other"
    otherCount: number;
    // Number of null points
    nullCount: number;
  };
}

export type AggregateKey = [number, number] | string;

export interface DistributionAggregate {
  select: Record<string, SQL.ExprNode>;
  collect: (row: any) => Record<string, any>;
  scales: Record<string, ScaleSpec>;
  order: Record<string, (a: AggregateKey, b: AggregateKey) => number>;
  clause: (values: Record<string, AggregateKey | AggregateKey[] | null | undefined>) => any;
}

/** Collect stats for distribution visualization.
 * For quantitative data, returns min, max, mean, median, and minPositive.
 * For nominal data, returns top-k levels and the corresponding count.
 * For non-supported data type, returns null. */
export async function distributionStats(
  coordinator: Coordinator,
  table: string,
  field: string,
): Promise<DistributionStats | null> {
  let query = (query: any): Promise<any> => coordinator.query(query);

  let desc = await query(SQL.Query.describe(SQL.Query.from(table).select({ field: SQL.column(field, table) })));
  let columnType = desc.get(0)?.column_type;
  if (columnType == null) {
    return null;
  }
  let jsColumnType = jsTypeFromDBType(columnType);
  if (jsColumnType == "number") {
    let fieldExpr = SQL.cast(SQL.column(field, table), "DOUBLE");
    let r1 = await query(
      SQL.Query.from(table)
        .select({
          count: SQL.count(),
          min: SQL.min(fieldExpr),
          minPositive: SQL.min(SQL.cond(SQL.gt(fieldExpr, 0), fieldExpr, SQL.literal(null))),
          max: SQL.max(fieldExpr),
          mean: SQL.avg(fieldExpr),
          median: SQL.median(fieldExpr),
        })
        .where(SQL.isFinite(fieldExpr)),
    );
    let r2 = await query(
      SQL.Query.from(table)
        .select({
          countNonFinite: SQL.count(),
        })
        .where(SQL.or(SQL.not(SQL.isFinite(fieldExpr)), SQL.isNull(fieldExpr))),
    );
    return {
      table: table,
      field: field,
      quantitative: { ...r1.get(0), ...r2.get(0) },
    };
  } else if (jsColumnType == "string") {
    let fieldExpr = SQL.cast(SQL.column(field, table), "TEXT");

    let levels: any[] = Array.from(
      await query(
        SQL.Query.from(table)
          .select({ value: fieldExpr, count: SQL.count() })
          .where(SQL.isNotNull(fieldExpr))
          .groupby(fieldExpr)
          .orderby(SQL.desc(SQL.count()))
          .limit(1000),
      ),
    );

    let nullCount: number = (
      await query(SQL.Query.from(table).select({ count: SQL.count() }).where(SQL.isNull(fieldExpr)))
    ).get(0).count;

    let { otherCount, numOtherLevels } = (
      await query(
        SQL.Query.from(table)
          .select({ otherCount: SQL.count(), numOtherLevels: SQL.sql`COUNT(DISTINCT(${fieldExpr}))` })
          .where(
            SQL.isNotNull(fieldExpr),
            SQL.not(
              SQL.isIn(
                fieldExpr,
                levels.map((x: any) => SQL.literal(x.value)),
              ),
            ),
          ),
      )
    ).get(0);

    return {
      table: table,
      field: field,
      nominal: {
        levels: levels,
        numOtherLevels: numOtherLevels,
        otherCount: otherCount,
        nullCount: nullCount,
      },
    };
  }
  return null;
}

export function distributionAggregate(
  ...fields: {
    key: string;
    stats: DistributionStats;
    scaleType?: ScaleType | null;
    binCount?: number;
  }[]
): DistributionAggregate {
  let select: Record<string, SQL.ExprNode> = {};
  let scales: Record<string, ScaleSpec> = {};
  let fieldMappers: Record<string, (v: any) => any> = {};
  let predicateMappers: Record<string, (v: AggregateKey | AggregateKey[]) => SQL.ExprNode | null> = {};
  let order: Record<string, (a: AggregateKey, b: AggregateKey) => number> = {};

  for (let field of fields) {
    // Quantitative data, infer binning
    if (field.stats.quantitative) {
      let binning = inferBinning(field.stats.quantitative!, {
        scale: field.scaleType,
        desiredCount: field.binCount ?? 20,
      });
      let inputExpr: SQL.ExprNode = SQL.cast(SQL.column(field.stats.field), "DOUBLE");
      let expr = binning.scale.expr(inputExpr, binning.scale.constant ?? 0);
      select[field.key] =
        binning.scale.type == "log"
          ? SQL.cond(
              SQL.and(SQL.isFinite(inputExpr), SQL.gt(inputExpr, SQL.literal(0))),
              SQL.floor(SQL.mul(SQL.sub(expr, binning.binStart), 1 / binning.binSize)),
              SQL.literal(null),
            )
          : SQL.cond(
              SQL.isFinite(inputExpr),
              SQL.floor(SQL.mul(SQL.sub(expr, binning.binStart), 1 / binning.binSize)),
              SQL.literal(null),
            );
      let valueToBinIndex = (x: number) => {
        return Math.floor((binning.scale.forward(x, binning.scale.constant ?? 0) - binning.binStart) / binning.binSize);
      };
      let binIndexToValue = (idx: number) => {
        return binning.scale.reverse(idx * binning.binSize + binning.binStart, binning.scale.constant ?? 0);
      };
      fieldMappers[field.key] = (v) => {
        if (v == null) {
          return "n/a";
        } else {
          return [binIndexToValue(v), binIndexToValue(v + 1)];
        }
      };
      let bin0 = valueToBinIndex(
        binning.scale.type == "log" ? field.stats.quantitative.minPositive : field.stats.quantitative.min,
      );
      let bin1 = valueToBinIndex(field.stats.quantitative.max);
      let domain = [binIndexToValue(bin0), binIndexToValue(bin1 + 1)];

      let hasNA = field.stats.quantitative.countNonFinite > 0;
      if (binning.scale.type == "log" && field.stats.quantitative.min < 0) {
        hasNA = true;
      }

      scales[field.key] = {
        type: binning.scale.type,
        constant: binning.scale.constant,
        domain: domain,
        specialValues: hasNA ? ["n/a"] : [],
      };

      let valueToPredicate = (v: AggregateKey | AggregateKey[]): SQL.ExprNode => {
        if (typeof v == "string") {
          if (v == "n/a") {
            return SQL.or(SQL.not(SQL.isFinite(inputExpr)), SQL.isNull(inputExpr));
          }
        } else if (v instanceof Array) {
          if (v.length == 2 && typeof v[0] == "number") {
            let [v1, v2] = v;
            if (typeof v1 == "number" && typeof v2 == "number") {
              return SQL.isBetween(inputExpr, [Math.min(v1, v2), Math.max(v1, v2)]);
            }
          } else {
            return SQL.or(...(v as AggregateKey[]).map(valueToPredicate));
          }
        }
        return SQL.literal(false);
      };

      predicateMappers[field.key] = valueToPredicate;

      order[field.key] = (a, b) => {
        let xa = typeof a == "string" ? [1, 0] : [0, a[0]];
        let xb = typeof b == "string" ? [1, 0] : [0, b[0]];
        if (xa[0] != xb[0]) {
          return xa[0] - xb[0];
        }
        return xa[1] - xb[1];
      };
    }
    // Nominal data, show top k levels and other/null if exists
    if (field.stats.nominal) {
      let binCount = field.binCount ?? 15;
      let { levels, nullCount, otherCount, numOtherLevels } = field.stats.nominal;
      if (levels.length > binCount) {
        // Clip to max binCount number of levels to display, combine others into "other" category
        numOtherLevels += levels.length - binCount;
        otherCount = levels.slice(binCount).reduce((a, b) => a + b.count, 0);
        levels = levels.slice(0, binCount);
      }
      let otherRepr = `(${numOtherLevels.toLocaleString()} others)`;
      let nullRepr = "(null)";

      let inputExpr: SQL.ExprNode = SQL.cast(SQL.column(field.stats.field), "TEXT");
      select[field.key] = SQL.cond(
        SQL.isIn(
          inputExpr,
          levels.map((l) => SQL.literal(l.value)),
        ),
        inputExpr,
        SQL.cond(SQL.isNull(inputExpr), SQL.literal(nullRepr), SQL.literal(otherRepr)),
      );

      let specialValues = [...(otherCount > 0 ? [otherRepr] : []), ...(nullCount > 0 ? [nullRepr] : [])];

      scales[field.key] = {
        type: "band",
        domain: levels.map((l) => l.value),
        specialValues: specialValues,
      };

      fieldMappers[field.key] = (v) => v;

      let predicate = (v: string) => {
        if (v == nullRepr) {
          return SQL.isNull(inputExpr);
        } else if (v == otherRepr) {
          return SQL.and(
            SQL.not(
              SQL.isIn(
                inputExpr,
                levels.map((l) => SQL.literal(l.value)),
              ),
            ),
            SQL.isNotNull(inputExpr),
          );
        } else {
          return SQL.isNotDistinct(inputExpr, SQL.literal(v));
        }
      };

      predicateMappers[field.key] = (v) => {
        if (v instanceof Array) {
          return SQL.or(...v.map((d) => predicate(d as string)));
        } else if (typeof v == "string") {
          return predicate(v);
        }
        return null;
      };

      let levelValues = levels.map((x) => x.value);

      order[field.key] = (a, b) => {
        if (typeof a == "string" && typeof b == "string") {
          let xa = levelValues.indexOf(a);
          if (xa < 0) {
            xa = levelValues.length + specialValues.indexOf(a);
          }
          let xb = levelValues.indexOf(b);
          if (xb < 0) {
            xb = levelValues.length + specialValues.indexOf(b);
          }
          return xa - xb;
        }
        return 0;
      };
    }
  }

  function collect(row: Record<string, any>): Record<string, any> {
    let result: Record<string, any> = { ...row };
    for (let field of fields) {
      result[field.key] = fieldMappers[field.key](result[field.key]);
    }
    return result;
  }

  function clause(values: Record<string, AggregateKey | AggregateKey[] | null | undefined>) {
    let parts: SQL.ExprNode[] = [];
    for (let field of fields) {
      let v = values[field.key];
      let part = v != null ? predicateMappers[field.key](v) : null;
      if (part) {
        parts.push(part);
      }
    }
    return {
      value: { ...values },
      predicate: parts.length > 0 ? SQL.and(...parts) : null,
    };
  }

  return { select, collect, scales, clause, order };
}
