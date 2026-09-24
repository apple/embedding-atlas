// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";

import type { FieldStats } from "../common/aggregate.js";
import { inferBinning } from "../common/binning.js";
import { inferNumberFormatter, inferTimeFormatter } from "../common/formatter.js";
import { continuousTicks } from "../common/ticks.js";
import { categoryIndexExpr, MAX_CATEGORIES, NULL_LABEL, OTHER_LABEL } from "./color.js";

/**
 * Fraction of a continuous axis reserved at the bottom for the "(null)" slot (when the field has
 * null / NaN / infinite values). The value range spans `[0, 1 - NULL_SLOT]`; nulls sit at `1`.
 */
const NULL_SLOT = 0.1;

/** Continuous scale types supported on an axis (numeric scales are auto-detected; time stays a time scale). */
type ContinuousScale = "linear" | "log" | "symlog" | "time";

export interface AxisTick {
  /** Position in normalized [0, 1] coordinates (0 = top). */
  value: number;
  label: string;
  /**
   * Label priority for overlap resolution (higher wins; default 0). When tick labels would overlap on
   * the Y axis the view keeps higher-priority labels and drops lower-priority ones. Continuous axes use
   * the tick level (decade ticks on log/symlog outrank intermediate ones); nominal axes use the category
   * frequency (more frequent categories outrank rarer ones and the "(other)" slot).
   */
  priority?: number;
}

export interface ResolvedAxis {
  title: string;
  /**
   * - `continuous`: `selectExpr` is the normalized FLOAT position in [0, 1] (0 = top).
   * - `nominal`: `selectExpr` is the integer category index; the view position is produced by jittering
   *   that index in JS (see applyCategoryJitter) using `categoryCount`.
   */
  kind: "continuous" | "nominal";
  selectExpr: SQL.ExprNode;
  /** Number of category slots (nominal only). */
  categoryCount?: number;
  ticks: AxisTick[];
  /** Build a SQL predicate for a normalized brush range; undefined if nothing is selected. */
  brushToPredicate: (from: number, to: number) => SQL.ExprNode | undefined;
}

/** Forward/inverse transform from data value to the scale's monotone space (the space ticks are even in). */
function scaleTransform(
  type: ContinuousScale,
  constant: number,
): { fwd: (x: number) => number; inv: (y: number) => number } {
  switch (type) {
    case "log":
      return { fwd: (x) => Math.log(x), inv: (y) => Math.exp(y) };
    case "symlog":
      return {
        fwd: (x) => Math.sign(x) * Math.log1p(Math.abs(x) / constant),
        inv: (y) => Math.sign(y) * Math.expm1(Math.abs(y)) * constant,
      };
    default: // linear, time (identity on the raw value / epoch ms)
      return { fwd: (x) => x, inv: (y) => y };
  }
}

/** SQL expression normalizing `raw` to [0,1] with 0 = top (= max), matching `scaleTransform`. */
function normalizeExpr(
  type: ContinuousScale,
  raw: SQL.ExprNode,
  tMax: number,
  tSpan: number,
  constant: number,
): SQL.ExprNode {
  switch (type) {
    case "log":
      // Only positive values are on a log axis; non-positive → NULL (the renderer skips non-finite rows).
      return SQL.sql`CASE WHEN ${raw} > 0 THEN (${SQL.literal(tMax)} - ln(${raw})) / ${SQL.literal(tSpan)} ELSE NULL END`;
    case "symlog":
      return SQL.sql`(${SQL.literal(tMax)} - sign(${raw}) * ln(1 + abs(${raw}) / ${SQL.literal(constant)})) / ${SQL.literal(tSpan)}`;
    default: // linear, time
      return SQL.sql`(${SQL.literal(tMax)} - ${raw}) / ${SQL.literal(tSpan)}`;
  }
}

/**
 * Resolve a single axis: a normalized [0,1] select expression (0 = top, mapping the field max / first
 * category to the top), tick labels, and a brush → predicate function. Supports quantitative, temporal,
 * and nominal fields. For numeric (non-time) fields the scale type (linear/log/symlog) is auto-detected the
 * same way as the chart runtime (via `inferBinning`). Returns null for unsupported / missing stats.
 */
export function resolveAxis(stats: FieldStats | undefined, expr: SQL.ExprNode, title: string): ResolvedAxis | null {
  if (stats == null) {
    return null;
  }

  if (stats.kind == "quantitative" || stats.kind == "temporal") {
    let isTemporal = stats.kind == "temporal";
    let raw = isTemporal ? SQL.epoch_ms(expr) : SQL.cast(expr, "DOUBLE");
    let s = isTemporal ? stats.temporal! : stats.quantitative!;
    let { min, max } = s;

    // Null, NaN and infinite values (and non-positive values on a log axis, see normalizeExpr) are
    // coalesced into a "(null)" slot at the bottom; the value range is compressed into [0, span].
    let hasNull = s.countNonFinite > 0;
    let span = hasNull ? 1 - NULL_SLOT : 1;
    let nullTick: AxisTick[] = hasNull ? [{ value: 1, label: NULL_LABEL, priority: 1 }] : [];
    // Map a normalized [0, 1] value position to the axis; rows where `check` is non-finite / NULL go to
    // the null slot (or NULL, which the renderer skips, when there is no slot).
    let withNullSlot = (normalized: SQL.ExprNode, check: SQL.ExprNode = normalized) =>
      SQL.cast(
        SQL.cond(
          SQL.isFinite(check),
          span == 1 ? normalized : SQL.mul(normalized, span),
          hasNull ? SQL.literal(1) : SQL.literal(null),
        ),
        "FLOAT",
      );
    let nullPredicate = (log: boolean) =>
      SQL.or(
        SQL.isNull(raw),
        SQL.not(SQL.isFinite(SQL.cast(raw, "DOUBLE"))),
        ...(log ? [SQL.not(SQL.gt(raw, 0))] : []),
      );

    // No finite values at all: only the null slot (if any).
    if (s.count == 0) {
      if (!hasNull) {
        return null;
      }
      return {
        title,
        kind: "continuous",
        selectExpr: SQL.cast(SQL.literal(1), "FLOAT"),
        ticks: nullTick,
        brushToPredicate: () => nullPredicate(false),
      };
    }

    // Degenerate (all equal / no finite range): draw a flat line in the middle of the value range,
    // no brushing on the value (the null slot can still be brushed).
    if (!(max > min)) {
      let mid = span / 2;
      // Format the single value (a date for temporal fields, not epoch ms).
      let label = isTemporal
        ? inferTimeFormatter([min], stats.temporal!.hasTimezone)(min)
        : inferNumberFormatter([min])(min);
      return {
        title,
        kind: "continuous",
        selectExpr: withNullSlot(SQL.literal(0.5), SQL.cast(raw, "DOUBLE")),
        ticks: [{ value: mid, label }, ...nullTick],
        brushToPredicate: (from: number, to: number) =>
          hasNull && Math.max(from, to) >= 1 - NULL_SLOT / 2 ? nullPredicate(false) : undefined,
      };
    }

    // Auto-detect the scale type for numeric fields (time stays a linear-in-ms time scale).
    let scaleType: ContinuousScale = "time";
    let constant = 1;
    if (!isTemporal) {
      let binning = inferBinning(stats.quantitative!, { desiredCount: 5 });
      scaleType = binning.scale.type;
      constant = binning.scale.constant ?? 1;
    }

    let { fwd, inv } = scaleTransform(scaleType, constant);
    let tMin = fwd(min);
    let tMax = fwd(max);
    let tSpan = tMax - tMin;
    if (!(tSpan > 0)) {
      // Transform collapsed the range (shouldn't happen for max>min); fall back to linear.
      scaleType = isTemporal ? "time" : "linear";
      fwd = (x) => x;
      inv = (y) => y;
      tMax = max;
      tSpan = max - min;
    }

    // value 0 = top = max.
    let selectExpr = withNullSlot(normalizeExpr(scaleType, raw, tMax, tSpan, constant));

    let tk = continuousTicks({
      type: scaleType,
      dataMin: min,
      dataMax: max,
      constant: constant,
      desiredCount: 5,
      extendDomainToTicks: false,
      hasTimezone: isTemporal ? stats.temporal!.hasTimezone : undefined,
    });
    let ticks: AxisTick[] = tk.values
      .filter((v) => v >= min && v <= max)
      // Lower tick levels are more important (level 0 = decade ticks on log/symlog); make them outrank
      // higher levels in label overlap resolution. Linear/time ticks are all level 0 (uniform priority).
      .map((v) => ({ value: ((tMax - fwd(v)) / tSpan) * span, label: tk.format(v), priority: -tk.level(v) }));
    ticks.push(...nullTick);

    let brushToPredicate = (from: number, to: number) => {
      let lo = Math.min(from, to);
      let hi = Math.max(from, to);
      let clauses: SQL.ExprNode[] = [];
      if (lo <= span) {
        // Axis position → normalized value position p → transform value (tMax - p*tSpan); smaller p
        // (top) → larger value.
        let pLo = lo / span;
        let pHi = Math.min(hi, span) / span;
        let dataHi = inv(tMax - pLo * tSpan);
        let dataLo = inv(tMax - pHi * tSpan);
        clauses.push(SQL.isBetween(raw, [Math.min(dataLo, dataHi), Math.max(dataLo, dataHi)]));
      }
      if (hasNull && hi >= 1 - NULL_SLOT / 2) {
        clauses.push(nullPredicate(scaleType == "log"));
      }
      if (clauses.length == 0) {
        return undefined;
      }
      return clauses.length == 1 ? clauses[0] : SQL.or(...clauses);
    };

    return { title, kind: "continuous", selectExpr, ticks, brushToPredicate };
  }

  // Nominal: top-k levels, then an "(other)" slot when there are more levels, then a "(null)" slot.
  let levels = stats.nominal!.levels.slice(0, MAX_CATEGORIES).map((l) => l.value);
  let k = levels.length;
  let hasOther = stats.nominal!.levels.length > k || stats.nominal!.numOtherLevels > 0 || stats.nominal!.otherCount > 0;
  let hasNull = stats.nominal!.nullCount > 0;
  let otherIndex = k;
  let nullIndex = k + (hasOther ? 1 : 0);
  let categoryCount = nullIndex + (hasNull ? 1 : 0);
  if (categoryCount == 0) {
    return null;
  }

  let textExpr = SQL.cast(expr, "TEXT");
  let selectExpr = categoryIndexExpr(textExpr, levels, otherIndex, nullIndex);

  // Tick at the center of each category's slot. More frequent categories get a higher label priority so
  // their labels survive overlap resolution; the "(other)" slot is least important (priority 0).
  let center = (i: number) => (i + 0.5) / categoryCount;
  let ticks: AxisTick[] = levels.map((v, i) => ({
    value: center(i),
    label: v,
    priority: stats.nominal!.levels[i].count,
  }));
  if (hasOther) {
    ticks.push({ value: center(otherIndex), label: OTHER_LABEL, priority: 0 });
  }
  if (hasNull) {
    ticks.push({ value: center(nullIndex), label: NULL_LABEL, priority: 0 });
  }

  let brushToPredicate = (from: number, to: number) => {
    let lo = Math.min(from, to);
    let hi = Math.max(from, to);
    let selectedLevels: string[] = [];
    let otherSelected = false;
    let nullSelected = false;
    // A category is selected when its slot overlaps the brush range.
    for (let c = 0; c < categoryCount; c++) {
      let slotLo = c / categoryCount;
      let slotHi = (c + 1) / categoryCount;
      if (slotLo <= hi + 1e-9 && slotHi >= lo - 1e-9) {
        if (c < k) {
          selectedLevels.push(levels[c]);
        } else if (hasOther && c == otherIndex) {
          otherSelected = true;
        } else {
          nullSelected = true;
        }
      }
    }
    let clauses: SQL.ExprNode[] = [];
    if (selectedLevels.length > 0) {
      clauses.push(
        SQL.isIn(
          textExpr,
          selectedLevels.map((v) => SQL.literal(v)),
        ),
      );
    }
    if (otherSelected) {
      // Everything non-null that is not in the top-k levels.
      clauses.push(
        k > 0
          ? SQL.and(
              SQL.isNotNull(textExpr),
              SQL.not(
                SQL.isIn(
                  textExpr,
                  levels.map((v) => SQL.literal(v)),
                ),
              ),
            )
          : SQL.isNotNull(textExpr),
      );
    }
    if (nullSelected) {
      clauses.push(SQL.isNull(textExpr));
    }
    if (clauses.length == 0) {
      return undefined;
    }
    return clauses.length == 1 ? clauses[0] : SQL.or(...clauses);
  };

  return { title, kind: "nominal", selectExpr, categoryCount, ticks, brushToPredicate };
}
