// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { jsTypeFromDBType } from "../../utils/database.js";
import { inferBinning } from "./binning.js";
import type { ScaleConfig, ScaleType } from "./types.js";

export interface FieldStats {
  field: SQL.ExprNode;
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

/** Collect stats for distribution visualization.
 * For quantitative data, returns min, max, mean, median, and minPositive.
 * For nominal data, returns top-k levels and the corresponding count.
 * For non-supported data type, returns null. */
export async function computeFieldStats(
  coordinator: Coordinator,
  table: SQL.FromExpr,
  field: SQL.ExprNode,
): Promise<FieldStats | undefined> {
  let query = (query: any): Promise<any> => coordinator.query(query);

  let desc = await query(SQL.Query.describe(SQL.Query.from(table).select({ field: field })));
  let columnType = desc.get(0)?.column_type;
  if (columnType == undefined) {
    return;
  }
  let jsColumnType = jsTypeFromDBType(columnType);
  if (jsColumnType == "number") {
    let fieldExpr = SQL.cast(field, "DOUBLE");
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
      field: field,
      quantitative: { ...r1.get(0), ...r2.get(0) },
    };
  } else if (jsColumnType == "string") {
    let fieldExpr = SQL.cast(field, "TEXT");

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
      field: field,
      nominal: {
        levels: levels,
        numOtherLevels: numOtherLevels,
        otherCount: otherCount,
        nullCount: nullCount,
      },
    };
  }
}

export type AggregateValue = [number, number] | string;

export interface AggregateInfo {
  select: SQL.ExprNode;
  scale: ScaleConfig;
  field: (v: any) => any;
  predicate: (v: AggregateValue | AggregateValue[]) => SQL.ExprNode | undefined;
  order: (a: AggregateValue, b: AggregateValue) => number;
}

export function inferAggregate({
  stats,
  scaleType,
  binCount,
}: {
  stats: FieldStats;
  scaleType?: ScaleType;
  binCount?: number;
}): AggregateInfo | undefined {
  // Quantitative data, infer binning
  if (stats.quantitative) {
    let binning = inferBinning(stats.quantitative!, {
      scale: scaleType,
      desiredCount: binCount ?? 20,
    });
    let inputExpr: SQL.ExprNode = SQL.cast(stats.field, "DOUBLE");
    let expr = binning.scale.expr(inputExpr, binning.scale.constant ?? 0);
    let select =
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
    let bin0 = valueToBinIndex(binning.scale.type == "log" ? stats.quantitative.minPositive : stats.quantitative.min);
    let bin1 = valueToBinIndex(stats.quantitative.max);
    let domain = [binIndexToValue(bin0), binIndexToValue(bin1 + 1)];

    let hasNA = stats.quantitative.countNonFinite > 0;
    if (binning.scale.type == "log" && stats.quantitative.min < 0) {
      hasNA = true;
    }

    let valueToPredicate = (v: AggregateValue | AggregateValue[]): SQL.ExprNode => {
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
          return SQL.or(...(v as AggregateValue[]).map(valueToPredicate));
        }
      }
      return SQL.literal(false);
    };

    return {
      select: select,
      scale: {
        type: binning.scale.type,
        constant: binning.scale.constant,
        domain: domain,
        specialValues: hasNA ? ["n/a"] : [],
      },
      predicate: valueToPredicate,
      order: (a, b) => {
        let xa = typeof a == "string" ? [1, 0] : [0, a[0]];
        let xb = typeof b == "string" ? [1, 0] : [0, b[0]];
        if (xa[0] != xb[0]) {
          return xa[0] - xb[0];
        }
        return xa[1] - xb[1];
      },
      field: (v) => {
        if (v == undefined) {
          return "n/a";
        } else {
          return [binIndexToValue(v), binIndexToValue(v + 1)];
        }
      },
    };
  }
  // Nominal data, show top k levels and other/null if exists
  if (stats.nominal) {
    binCount = binCount ?? 15;
    let { levels, nullCount, otherCount, numOtherLevels } = stats.nominal;
    if (levels.length > binCount) {
      // Clip to max binCount number of levels to display, combine others into "other" category
      numOtherLevels += levels.length - binCount;
      otherCount = levels.slice(binCount).reduce((a, b) => a + b.count, 0);
      levels = levels.slice(0, binCount);
    }
    let otherRepr = `(${numOtherLevels.toLocaleString()} others)`;
    let nullRepr = "(null)";

    let inputExpr: SQL.ExprNode = SQL.cast(stats.field, "TEXT");
    let select = SQL.cond(
      SQL.isIn(
        inputExpr,
        levels.map((l) => SQL.literal(l.value)),
      ),
      inputExpr,
      SQL.cond(SQL.isNull(inputExpr), SQL.literal(nullRepr), SQL.literal(otherRepr)),
    );

    let specialValues = [...(otherCount > 0 ? [otherRepr] : []), ...(nullCount > 0 ? [nullRepr] : [])];

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

    let levelValues = levels.map((x) => x.value);

    return {
      select: select,
      scale: {
        type: "band",
        domain: levels.map((l) => l.value),
        specialValues: specialValues,
      },
      field: (v) => v,
      predicate: (v) => {
        if (v instanceof Array) {
          return SQL.or(...v.map((d) => predicate(d as string)));
        } else if (typeof v == "string") {
          return predicate(v);
        }
      },
      order: (a, b) => {
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
      },
    };
  }
}
