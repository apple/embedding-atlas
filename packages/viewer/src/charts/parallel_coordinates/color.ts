// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as SQL from "@uwdata/mosaic-sql";
import * as d3 from "d3";

import type { FieldStats } from "../common/aggregate.js";
import { inferBinning } from "../common/binning.js";
import { inferColorScale } from "../common/infer.js";
import { resolveInterpolate, type ChartTheme } from "../common/theme.js";
import type { ConcreteScale, ScaleConfig } from "../common/types.js";
import type { Scale } from "../spec/spec.js";

/** Maximum number of discrete categories shown for a categorical color field; the rest become "(other)". */
export const MAX_CATEGORIES = 10;

/** Label of the slot collecting the levels beyond the top-k. */
export const OTHER_LABEL = "(other)";

/** Label of the slot collecting null / NaN / infinite values. */
export const NULL_LABEL = "(null)";

/**
 * Size of the renderer's color LUT. A continuous color scheme is sampled into this many stops; when the
 * field has nulls the last entry is the null color instead (and the scheme gets one stop fewer).
 */
const CONTINUOUS_STOPS = 256;

/**
 * SQL expression mapping `textExpr` to an integer category index: `i` for `levels[i]`, `nullIndex` for
 * NULL, `otherIndex` for anything else.
 */
export function categoryIndexExpr(
  textExpr: SQL.ExprNode,
  levels: string[],
  otherIndex: number,
  nullIndex: number,
): SQL.ExprNode {
  let caseExpr = SQL.sql`CASE WHEN ${textExpr} IS NULL THEN ${SQL.literal(nullIndex)} ${levels
    .map((v, i) => SQL.sql`WHEN ${textExpr} = ${SQL.literal(v)} THEN ${SQL.literal(i)}`)
    .join(" ")} ELSE ${SQL.literal(otherIndex)} END`;
  return SQL.cast(caseExpr, "UTINYINT");
}

export type ColorKind = "none" | "continuous" | "categorical";

export interface ResolvedColorData {
  kind: ColorKind;
  /** The SQL expression selected per row (raw value for continuous, integer index for categorical, 0 for none). */
  selectExpr: SQL.ExprNode;
  /** Convert the queried arrow column into the renderer's colorValue (Float32 for continuous, Uint8 index otherwise). */
  toColorValue: (column: any, numRows: number) => Float32Array | Uint8Array;
  /** For categorical: the display level labels (top-k, then "(other)" and "(null)" when present). */
  levels?: string[];
  /** For continuous: whether some rows are null (drawn with the null color, the last palette entry). */
  hasNull?: boolean;
  /**
   * For continuous: a theme-independent scale config (type + [lo, hi] domain) used to render the color
   * legend. `buildColorScale` turns this into a concrete value→color scale via the theme.
   */
  legendScaleConfig?: ScaleConfig;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** A function mapping a data value to a normalized [0, 1] position, per the given scale type. */
function makeNormalizer(type: Scale["type"], lo: number, hi: number, constant?: number): (v: number) => number {
  if (!(hi > lo)) {
    hi = lo + 1;
  }
  switch (type) {
    case "log": {
      let scale = d3
        .scaleLog()
        .domain([lo > 0 ? lo : 1e-9, hi > 0 ? hi : 1])
        .range([0, 1])
        .clamp(true);
      return (v) => clamp01(scale(v > 0 ? v : 1e-12));
    }
    case "symlog": {
      let scale = d3
        .scaleSymlog()
        .constant(constant ?? 1)
        .domain([lo, hi])
        .range([0, 1])
        .clamp(true);
      return (v) => clamp01(scale(v));
    }
    default: {
      let scale = d3.scaleLinear().domain([lo, hi]).range([0, 1]).clamp(true);
      return (v) => clamp01(scale(v));
    }
  }
}

/**
 * Resolve the per-row color encoding (query side only — independent of theme). Continuous for
 * quantitative/temporal fields, categorical for nominal fields (capped at MAX_CATEGORIES + "(other)").
 */
export function resolveColorData(
  stats: FieldStats | undefined,
  scaleSpec: Scale | undefined,
  expr: SQL.ExprNode | null,
): ResolvedColorData {
  if (expr == null || stats == null) {
    return {
      kind: "none",
      selectExpr: SQL.literal(0),
      toColorValue: (_col, n) => new Uint8Array(n),
    };
  }

  // Categorical: top-k levels (+ "(other)", + "(null)").
  if (stats.kind == "nominal") {
    let levels = stats.nominal!.levels.slice(0, MAX_CATEGORIES).map((l) => l.value);
    let hasOther =
      stats.nominal!.levels.length > levels.length ||
      stats.nominal!.numOtherLevels > 0 ||
      stats.nominal!.otherCount > 0;
    let hasNull = stats.nominal!.nullCount > 0;
    let otherIndex = levels.length;
    let nullIndex = otherIndex + (hasOther ? 1 : 0);
    return {
      kind: "categorical",
      selectExpr: categoryIndexExpr(SQL.cast(expr, "TEXT"), levels, otherIndex, nullIndex),
      toColorValue: (col) => {
        let arr = col.toArray();
        return arr instanceof Uint8Array ? arr : Uint8Array.from(arr, (v: any) => (v == null ? nullIndex : Number(v)));
      },
      levels: [...levels, ...(hasOther ? [OTHER_LABEL] : []), ...(hasNull ? [NULL_LABEL] : [])],
    };
  }

  // Continuous: quantitative or temporal.
  let isTemporal = stats.kind == "temporal";
  let raw = isTemporal ? SQL.epoch_ms(expr) : SQL.cast(expr, "DOUBLE");
  let dataMin = isTemporal ? stats.temporal!.min : stats.quantitative!.min;
  let dataMax = isTemporal ? stats.temporal!.max : stats.quantitative!.max;
  let domain = (scaleSpec?.domain as number[] | undefined) ?? undefined;
  let lo = typeof domain?.[0] == "number" ? (domain[0] as number) : dataMin;
  let hi = typeof domain?.[1] == "number" ? (domain[1] as number) : dataMax;

  // Infer the scale type the same way the axes and the embedding view's color do: auto-detect
  // linear/log/symlog from the data via `inferBinning` (unless the spec pins a type). Temporal stays a
  // linear-in-epoch-ms "time" scale. This keeps a field's color ramp consistent with its axis.
  let scaleType: Scale["type"];
  let constant = scaleSpec?.constant;
  if (scaleSpec?.type != null) {
    scaleType = scaleSpec.type;
  } else if (isTemporal) {
    scaleType = "time";
  } else {
    let binning = inferBinning(stats.quantitative!, { desiredCount: 5 });
    scaleType = binning.scale.type;
    constant = constant ?? binning.scale.constant ?? 1;
  }

  let normalize = makeNormalizer(scaleType, lo, hi, constant);
  // Null / NaN / infinite values (and non-positive values on a log scale) are drawn with the null color:
  // the scheme is squeezed into [0, rampEnd] and nulls map to 1 (the last LUT entry).
  let isLog = scaleType == "log";
  let hasNull = (isTemporal ? stats.temporal! : stats.quantitative!).countNonFinite > 0 || (isLog && dataMin <= 0);
  let rampEnd = hasNull ? (CONTINUOUS_STOPS - 2) / (CONTINUOUS_STOPS - 1) : 1;
  return {
    kind: "continuous",
    selectExpr: raw,
    toColorValue: (col, n) => {
      let arr = col.toArray();
      let out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let x = arr[i];
        let v = x == null ? NaN : Number(x);
        out[i] = isFinite(v) && !(isLog && v <= 0) ? normalize(v) * rampEnd : hasNull ? 1 : 0;
      }
      return out;
    },
    hasNull,
    // The legend uses the same inferred scale type so the ramp matches what the ribbons encode.
    legendScaleConfig: {
      type: scaleType,
      domain: [lo, hi],
      constant,
      range: scaleSpec?.range,
      specialValues: hasNull ? [NULL_LABEL] : undefined,
    },
  };
}

/** Build the color palette (theme-dependent) for the resolved color encoding. */
export function buildPalette(color: ResolvedColorData, range: Scale["range"] | undefined, theme: ChartTheme): string[] {
  if (color.kind == "none") {
    return [theme.embeddingColor];
  }

  if (color.kind == "categorical") {
    let labels = color.levels ?? [];
    let baseCount = labels.filter((l) => l != OTHER_LABEL && l != NULL_LABEL).length;
    let base: string[];
    if (Array.isArray(range) && range.length > 0) {
      base = range.map((c) => String(c));
    } else if (typeof theme.categoryColors == "function") {
      base = theme.categoryColors(Math.max(1, baseCount));
    } else {
      base = theme.categoryColors;
    }
    return labels.map((l, i) =>
      l == OTHER_LABEL ? theme.otherColor : l == NULL_LABEL ? theme.nullColor : base[i % base.length],
    );
  }

  // Continuous: sample the interpolate scheme into stops; the renderer interpolates across them.
  let interp = resolveInterpolate((range as string | string[] | undefined) ?? theme.interpolate);
  let stops = color.hasNull ? CONTINUOUS_STOPS - 1 : CONTINUOUS_STOPS;
  let out: string[] = [];
  for (let i = 0; i < stops; i++) {
    out.push(interp(i / (stops - 1)));
  }
  if (color.hasNull) {
    out.push(theme.nullColor);
  }
  return out;
}

/**
 * Build a concrete value→color scale for the color legend, or `null` when there's nothing to show
 * (single-color / no encoding). The categorical scale is built straight from `levels` + `palette` so
 * the legend swatches (including "(other)") match the rendered ribbons exactly; the continuous scale
 * reuses `inferColorScale`, which interpolates the same scheme as `buildPalette`.
 */
export function buildColorScale(
  color: ResolvedColorData,
  palette: string[],
  theme: ChartTheme,
): ConcreteScale<string> | null {
  if (color.kind == "categorical") {
    let levels = color.levels ?? [];
    if (levels.length == 0) {
      return null;
    }
    let isSpecial = (l: string) => l == OTHER_LABEL || l == NULL_LABEL;
    let map = new Map<string, string>();
    levels.forEach((label, i) => map.set(label, palette[i] ?? theme.markColorGray));
    return {
      type: "band",
      domain: levels.filter((l) => !isSpecial(l)),
      specialValues: levels.filter(isSpecial),
      apply: (value) => map.get(value) ?? theme.markColorGray,
    };
  }
  if (color.kind == "continuous" && color.legendScaleConfig != null) {
    let scale = inferColorScale(color.legendScaleConfig, theme);
    if (!color.hasNull) {
      return scale;
    }
    // Match the null color the ribbons are drawn with (the last palette entry).
    let nullColor = palette[palette.length - 1];
    return { ...scale, apply: (value) => (value == NULL_LABEL ? nullColor : scale.apply(value)) };
  }
  return null;
}
