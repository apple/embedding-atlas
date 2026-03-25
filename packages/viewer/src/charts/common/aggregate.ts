// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { inferBinning, inferTimeBinning, type Binning } from "./binning.js";
import type { ScaleConfig, ScaleType } from "./types.js";

export interface FieldStats {
  field: SQL.ExprNode;
  kind: "quantitative" | "temporal" | "nominal";
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
  temporal?: {
    /** Number of finite values */
    count: number;
    /** The minimum finite value in seconds since epoch */
    min: number;
    /** The maximum finite value in seconds since epoch */
    max: number;

    /** Number of non-finite values (inf, nan, null) */
    countNonFinite: number;

    /**
     * Whether the data includes timezone information.
     * The field is always casted to milliseconds since epoch (1970-01-01 00:00:00+00).
     * If true, the result is a true UTC time. We'll display the timestamp in the current timezone.
     * If false, the result may or may not be a true UTC time because timezone information is missing.
     * We will display the timestamp in the UTC timezone, but without any timezone indicator.
     */
    hasTimezone: boolean;
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

const quantitativeTypes = new Set([
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

const temporalTypes = new Set([
  "DATE",
  "TIME",
  "DATETIME",
  "TIMESTAMP",
  "TIMESTAMPTZ",
  "TIMESTAMP WITH TIME ZONE",
  "TIMESTAMP WITHOUT TIME ZONE",
]);

const temporalTypesWithTimezone = new Set(["TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"]);

const nominalTypes = new Set(["BOOLEAN", "DATE", "VARCHAR", "CHAR", "BPCHAR", "TEXT", "STRING"]);

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

  // Quantitative types
  if (quantitativeTypes.has(columnType)) {
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
      kind: "quantitative",
      quantitative: { ...r1.get(0), ...r2.get(0) },
    };
  }

  // Temporal types
  if (temporalTypes.has(columnType)) {
    let hasTimezone = temporalTypesWithTimezone.has(columnType);
    let fieldExpr = SQL.epoch_ms(field);
    let r1 = await query(
      SQL.Query.from(table)
        .select({
          count: SQL.count(),
          min: SQL.min(fieldExpr),
          max: SQL.max(fieldExpr),
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
      kind: "temporal",
      temporal: {
        ...r1.get(0),
        ...r2.get(0),
        hasTimezone: hasTimezone,
      },
    };
  }

  // Nominal types
  if (nominalTypes.has(columnType)) {
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
      kind: "nominal",
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
  predicate: (v: AggregateValue | AggregateValue[]) => SQL.ExprNode;
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
  // Quantitative or temporal data, infer binning
  if (stats.quantitative || stats.temporal) {
    let binning: Binning;
    let inputExpr: SQL.ExprNode;
    let hasNA = false;

    if (stats.quantitative) {
      binning = inferBinning(stats.quantitative, {
        scale: scaleType,
        desiredCount: binCount ?? 20,
      });
      hasNA = stats.quantitative.countNonFinite > 0;
      inputExpr = SQL.cast(stats.field, "DOUBLE");
    } else if (stats.temporal) {
      binning = inferTimeBinning(stats.temporal, {
        desiredCount: binCount ?? 20,
        hasTimezone: stats.temporal.hasTimezone,
      });
      hasNA = stats.temporal.countNonFinite > 0;
      inputExpr = SQL.epoch_ms(stats.field);
    } else {
      throw new Error("invalid stats");
    }

    let select = binning.binIndexExpr(inputExpr);

    // For log scale, if we have <= values, we have n/a.
    if (stats.quantitative && binning.scale.type == "log" && stats.quantitative.min <= 0) {
      hasNA = true;
    }

    let valueToPredicate = (v: AggregateValue | AggregateValue[]): SQL.ExprNode => {
      if (typeof v == "string") {
        if (v == "n/a") {
          return binning.nullPredicateExpr(inputExpr);
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
        domain: binning.scale.domain,
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
          return binning.rangeForIndex(v);
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
        } else {
          return predicate(v.toString());
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
