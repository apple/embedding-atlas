// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { categoryIndexExpr, MAX_SCALE_CATEGORIES, type ColorScale } from "@embedding-atlas/component";
import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";

import { distinctCountExpr, jsTypeFromDBType } from "../../utils/database.js";
import { computeFieldStats } from "../common/aggregate.js";
import { inferBinning, inferTimeBinning, type Binning } from "../common/binning.js";
import { inferNumberFormatter, inferTimeFormatter } from "../common/formatter.js";
import { resolveInterpolate, type ChartTheme } from "../common/theme.js";
import type { EmbeddingColorScale } from "./types.js";

/** The number of categories by frequency when a categorical scale has no domain. */
export const DEFAULT_MAX_CATEGORIES = 10;
/** The desired number of bins of a binned scale. */
export const DEFAULT_BIN_COUNT = 5;

/** The categories of a category field (independent of the theme): which values are which category, and their
 *  labels and counts (of all data). Colors are added with `resolveLegend`. */
export interface LegendData {
  kind: "categorical" | "binned";
  /** The scale's categories: a `domain`, or `thresholds` (without colors). */
  scale: ColorScale;
  /** The labels of the categories (or bins). */
  labels: string[];
  /** The counts of the categories (or bins), then of "other" and "null" (see `categoryIndexExpr`). */
  counts: number[];
  /** The label of the "other" category. */
  otherLabel: string;
}

export interface EmbeddingLegend {
  /** The scale for the embedding view: the categories and their colors. */
  scale: ColorScale;
  items: { label: string; color: string; predicate: any; count: number }[];
}

/** The categories of a field: values (compared as text) for strings and booleans, bins for numbers and times
 *  (as in the charts; to color numbers as categories, cast them to text). Returns null if the field's type isn't
 *  supported. */
export async function computeLegendData(
  coordinator: Coordinator,
  table: string,
  field: SQL.ExprNode,
  spec: EmbeddingColorScale | null | undefined,
): Promise<LegendData | null> {
  let kind = await inferKind(coordinator, table, field);
  switch (kind) {
    case "categorical":
      return await categoricalData(coordinator, table, field, spec);
    case "number":
    case "time":
      return await binnedData(coordinator, table, field, kind, spec);
    default:
      return null;
  }
}

async function inferKind(
  coordinator: Coordinator,
  table: string,
  field: SQL.ExprNode,
): Promise<"categorical" | "number" | "time" | null> {
  let [desc] = Array.from(
    (await coordinator.query(SQL.Query.describe(SQL.Query.from(table).select({ field })))) as any,
  ) as any[];
  if (desc == null) {
    return null;
  }
  let dbType: string = desc.column_type;
  let jsType = jsTypeFromDBType(dbType);
  if (jsType == "string" || dbType == "BOOLEAN") {
    return "categorical";
  }
  if (jsType == "number") {
    return "number";
  }
  if (jsType == "Date") {
    return "time";
  }
  return null;
}

/** Count rows by category index (k categories, then other and null). */
async function countByIndex(
  coordinator: Coordinator,
  table: string,
  indexExpr: SQL.ExprNode,
  k: number,
): Promise<number[]> {
  let rows = Array.from(
    (await coordinator.query(
      SQL.Query.from(table)
        .select({ index: indexExpr, count: SQL.cast(SQL.count(), "INT") })
        .groupby(indexExpr),
    )) as any,
  ) as { index: number; count: number }[];
  let counts = new Array(k + 2).fill(0);
  for (let { index, count } of rows) {
    if (index != null && index >= 0 && index < k + 2) {
      counts[Number(index)] = count;
    }
  }
  return counts;
}

async function categoricalData(
  coordinator: Coordinator,
  table: string,
  field: SQL.ExprNode,
  spec: EmbeddingColorScale | null | undefined,
): Promise<LegendData> {
  let domain: string[];
  if (spec?.domain != null) {
    domain = spec.domain.map((v) => String(v));
  } else {
    // The most frequent values.
    let text = SQL.cast(field, "TEXT");
    let rows = Array.from(
      (await coordinator.query(
        SQL.Query.from(table)
          .select({ value: text, count: SQL.count() })
          .where(SQL.isNotNull(text))
          .groupby(text)
          .orderby(SQL.desc(SQL.count()), text)
          .limit(DEFAULT_MAX_CATEGORIES),
      )) as any,
    ) as { value: string }[];
    domain = rows.map((r) => r.value);
  }
  domain = domain.slice(0, MAX_SCALE_CATEGORIES);
  let scale: ColorScale = { domain };
  let k = domain.length;
  let indexExpr = categoryIndexExpr(field, scale);
  let counts = await countByIndex(coordinator, table, indexExpr, k);
  let otherLabel = "(other)";
  if (counts[k] > 0) {
    let n = await distinctCountExpr(coordinator, table, SQL.cast(field, "TEXT"), SQL.eq(indexExpr, SQL.literal(k)));
    otherLabel = `(${n.toLocaleString()} ${n == 1 ? "other" : "others"})`;
  }
  return { kind: "categorical", scale, labels: domain, counts, otherLabel };
}

/** Parse a domain value of a time scale (epoch milliseconds, or a date string). */
function timeValue(v: unknown): number | null {
  if (typeof v == "number") {
    return v;
  }
  if (typeof v == "string") {
    let t = Date.parse(v);
    return isNaN(t) ? null : t;
  }
  return null;
}

async function binnedData(
  coordinator: Coordinator,
  table: string,
  field: SQL.ExprNode,
  kind: "number" | "time",
  spec: EmbeddingColorScale | null | undefined,
): Promise<LegendData | null> {
  // Numbers are binned as doubles (so that, e.g., DECIMAL and HUGEINT fields have quantitative stats).
  let stats = await computeFieldStats(coordinator, table, kind == "number" ? SQL.cast(field, "DOUBLE") : field);
  let desiredCount = spec?.bin?.desiredCount ?? DEFAULT_BIN_COUNT;

  let formatter: (values: number[]) => (v: number) => string;
  let type: "number" | "time";
  // The range of values to bin ([lo, hi]), and whether it's an explicit domain (values outside it are "other").
  let lo: number | null = null;
  let hi: number | null = null;
  let explicit = false;
  let infer: (lo: number, hi: number) => Binning;
  // The bin of a constant field (all values equal to v).
  let constantBin: (v: number) => [number, number];
  if (stats?.quantitative != null) {
    type = "number";
    let q = stats.quantitative;
    let [dlo, dhi] = spec?.domain ?? [];
    explicit = typeof dlo == "number" && typeof dhi == "number" && dhi >= dlo;
    if (explicit) {
      [lo, hi] = [dlo as number, dhi as number];
    } else if (q.count > 0) {
      [lo, hi] = [q.min, q.max];
    }
    let scale = spec?.type ?? null;
    if (scale == "log" && lo != null && hi != null && lo <= 0) {
      // Zero and negative values can't be on a log scale: they're "other", and the bins cover the positive values.
      let minPositive = q.minPositive;
      lo = minPositive != null && isFinite(minPositive) && minPositive <= hi ? minPositive : null;
    }
    infer = (lo, hi) => inferBinning({ ...q, min: lo, max: hi, minPositive: lo }, { scale, desiredCount });
    constantBin = (v) => (scale == "log" ? [v / 2, v * 2] : [v - 1, v + 1]);
    formatter = inferNumberFormatter;
  } else if (stats?.temporal != null) {
    type = "time";
    let t = stats.temporal;
    let dlo = timeValue(spec?.domain?.[0]);
    let dhi = timeValue(spec?.domain?.[1]);
    explicit = dlo != null && dhi != null && dhi >= dlo;
    if (explicit) {
      [lo, hi] = [dlo, dhi];
    } else if (t.count > 0) {
      [lo, hi] = [t.min, t.max];
    }
    infer = (lo, hi) => inferTimeBinning({ ...t, min: lo, max: hi }, { desiredCount, hasTimezone: t.hasTimezone });
    const day = 24 * 3600 * 1000;
    constantBin = (v) => [v - day, v + day];
    formatter = (v) => inferTimeFormatter(v, t.hasTimezone);
  } else {
    return null;
  }

  // The bins covering [lo, hi] (the upper bound of the last bin is inclusive).
  let thresholds: number[] = [];
  if (lo != null && hi != null && isFinite(lo) && isFinite(hi)) {
    if (lo == hi) {
      // A single value: an explicit domain is just that value, otherwise a bin around it.
      thresholds = explicit ? [lo, hi] : constantBin(lo);
    } else {
      let binning = infer(lo, hi);
      let first = binning.binIndex(lo);
      let last = binning.binIndex(hi);
      if (explicit) {
        // The domain's bounds are the ends of the first and last bins. Bin boundaries are computed in floating
        // point, so a bound at a boundary may land in the adjacent bin (e.g., 0.9 in a bin starting at
        // 0.8999999999999999): don't add a bin that only touches a bound. (Without a domain, the bins are the
        // charts' bins: the max may start a bin of its own, e.g., 4 in [4, 5) for integers 0 to 4.)
        let touches = (index: number, value: number, side: 0 | 1) => {
          let [start, end] = binning.rangeForIndex(index);
          return Math.abs((side == 0 ? start : end) - value) <= Math.abs(end - start) * 1e-9;
        };
        if (last > first && (binning.rangeForIndex(last)[0] >= hi || touches(last, hi, 0))) {
          last -= 1;
        }
        if (last > first && touches(first, lo, 1)) {
          first += 1;
        }
      }
      if (isFinite(first) && isFinite(last) && last >= first) {
        last = Math.min(last, first + MAX_SCALE_CATEGORIES - 1);
        for (let i = first; i <= last; i++) {
          thresholds.push(binning.rangeForIndex(i)[0]);
        }
        thresholds.push(binning.rangeForIndex(last)[1]);
        let n = thresholds.length - 1;
        if (explicit) {
          // Values outside the domain are "other".
          thresholds[0] = lo;
          thresholds[n] = hi;
        } else {
          // The bins cover all values (bin boundaries may be slightly off in floating point).
          thresholds[0] = Math.min(thresholds[0], lo);
          thresholds[n] = Math.max(thresholds[n], hi);
        }
      }
    }
  }

  let scale: ColorScale = { thresholds, type };
  let k = Math.max(thresholds.length - 1, 0);
  let format = formatter(thresholds);
  let labels: string[] = [];
  for (let i = 0; i < k; i++) {
    let close = i == k - 1 ? "]" : ")";
    labels.push(`[${format(thresholds[i])}, ${format(thresholds[i + 1])}${close}`);
  }
  let counts = await countByIndex(coordinator, table, categoryIndexExpr(field, scale), k);
  return { kind: "binned", scale, labels, counts, otherLabel: "(other)" };
}

/** The colors of the categories: categorical fields use the range (a list of colors, or an interpolate) or the
 *  theme's category colors; binned fields sample the range (an interpolate name, or colors to interpolate) or
 *  the theme's continuous interpolate. */
function categoryColors(data: LegendData, spec: EmbeddingColorScale | null | undefined, theme: ChartTheme): string[] {
  let k = data.labels.length;
  let range = spec?.range;
  if (data.kind == "categorical" && range == null) {
    let colors =
      typeof theme.categoryColors == "function" ? theme.categoryColors(Math.max(k, 1)) : theme.categoryColors;
    return data.labels.map((_, i) => colors[i % colors.length]);
  }
  if (data.kind == "categorical" && Array.isArray(range) && range.length > 0) {
    return data.labels.map((_, i) => range[i % range.length]);
  }
  let interpolate = resolveInterpolate(
    typeof range == "string" || (Array.isArray(range) && range.length > 0) ? range : theme.interpolate,
  );
  return data.labels.map((_, i) => interpolate((i + 0.5) / k));
}

/** The legend of a field: the view's scale (categories and colors), and the legend items with their predicates
 *  (categories without rows aren't listed, except for an explicit domain and bins). */
export function resolveLegend(
  data: LegendData,
  field: SQL.ExprNode,
  spec: EmbeddingColorScale | null | undefined,
  theme: ChartTheme,
): EmbeddingLegend {
  let colors = categoryColors(data, spec, theme);
  let scale: ColorScale = {
    ...data.scale,
    range: colors,
    other: spec?.other ?? theme.otherColor,
    null: spec?.null ?? theme.nullColor,
  };
  let indexExpr = categoryIndexExpr(field, scale);
  let k = colors.length;
  let item = (index: number, label: string, color: string) => ({
    label,
    color,
    predicate: SQL.eq(indexExpr, SQL.literal(index)),
    count: data.counts[index] ?? 0,
  });
  let items = data.labels.map((label, i) => item(i, label, colors[i]));
  if (data.counts[k] > 0) {
    items.push(item(k, data.otherLabel, scale.other!));
  }
  if (data.counts[k + 1] > 0) {
    items.push(item(k + 1, "(null)", scale.null!));
  }
  return { scale, items };
}
