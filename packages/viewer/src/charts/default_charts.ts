// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";

import { columnDescriptions, distinctCountBatch, isFloatingPointDBType, type ColumnDesc } from "../utils/database.js";
import type { BuiltinChartSpec } from "./chart_types.js";
import type { EmbeddingSpec } from "./embedding/types.js";
import type { FeaturesListSpec } from "./features/types.js";
import type { InstancesSpec } from "./instances/types.js";
import type { ChartSpec } from "./spec/spec.js";

export interface DefaultChartsConfig {
  /** If specified, only include the given columns */
  include?: string[];

  /** Columns to exclude, applicable if `include` is not specified */
  exclude?: string[];

  /** Override the chart spec for certain columns. If the override is set to `null` the column will be skipped */
  override?: Record<string, BuiltinChartSpec | null>;

  /** Set to false to disable the instances table, or an object to override spec properties */
  table?: boolean | Partial<InstancesSpec>;

  /** Set to false to disable the embedding view, or an object to override spec properties */
  embedding?: boolean | Partial<EmbeddingSpec>;
}

interface ProjectionInput {
  x: string;
  y: string;
  text?: string;
  isGis?: boolean;
  image?: string;
  importance?: string;
  neighbors?: string;
  bounds?: { x: [number, number]; y: [number, number] } | null;
  /** Names of pre-computed u16-quantised x/y columns. When set, the
   *  scatter wire query becomes a pure scan — see `EmbeddingViewMosaic`. */
  precomputed?: { x_u16: string; y_u16: string; y_is_mercator?: boolean } | null;
  /** Viewport defaults from server-side bbox knowledge. See
   *  `EmbeddingSpec.data.viewportHint`. */
  viewportHint?: {
    centerX: number;
    centerY: number;
    rangeX: number;
    rangeY: number;
    rowCount?: number;
    skipDeferredRefine?: boolean;
  } | null;
}

/** Synchronous chart specs that need no DB queries: embedding, predicates,
 *  and the instances table. Mount these *first* so the embedding view's own
 *  density+scatter queries can race in parallel with the slower
 *  per-column distinct-count discovery. On 75 M-row datasets, the wide
 *  APPROX_COUNT_DISTINCT batch takes 5–10 s, and gating the embedding mount
 *  on it shifted first-paint by exactly that much. */
export function defaultPrimaryCharts(options: {
  projection?: ProjectionInput;
  features?: string;
  config?: DefaultChartsConfig;
}): BuiltinChartSpec[] {
  let { projection, features } = options;
  let config = options.config ?? {};
  let charts: BuiltinChartSpec[] = [];

  if (projection != null && config.embedding !== false) {
    let spec: EmbeddingSpec = {
      type: "embedding",
      title: "Embedding",
      data: {
        x: projection.x,
        y: projection.y,
        text: projection.text,
        isGis: projection.isGis,
        image: projection.image,
        importance: projection.importance,
        bounds: projection.bounds,
        precomputed: projection.precomputed,
        viewportHint: projection.viewportHint,
        neighbors: projection.neighbors,
      },
    };
    if (typeof config.embedding == "object") {
      spec = { ...spec, ...config.embedding };
    }
    charts.push(spec);
  }

  if (features != null) {
    let spec: FeaturesListSpec = {
      type: "features-list",
      title: "Features",
      data: {
        features: features,
      },
    };
    charts.push(spec);
  }

  charts.push({ type: "predicates", title: "SQL Predicates" });

  if (config.table !== false) {
    let spec: InstancesSpec = {
      type: "instances",
      title: "Instances",
    };
    if (typeof config.table == "object") {
      spec = { ...spec, ...config.table };
    }
    charts.push(spec);
  }

  return charts;
}

/** Async per-column chart discovery. Issues one fused
 *  ``APPROX_COUNT_DISTINCT`` query and returns a histogram or count-plot
 *  per qualifying column. Caller is expected to merge these into the chart
 *  set after the primary charts have already mounted. */
export async function defaultColumnCharts(options: {
  coordinator: Coordinator;
  table: string;
  columns?: ColumnDesc[];
  projection?: ProjectionInput;
  features?: string;
  config?: DefaultChartsConfig;
}): Promise<BuiltinChartSpec[]> {
  let { coordinator, table, projection } = options;
  let config = options.config ?? {};
  let exclude = [...(config.exclude ?? [])];
  if (projection != null && config.embedding !== false) {
    exclude.push(projection.x);
    exclude.push(projection.y);
    if (projection.text) {
      exclude.push(projection.text);
    }
  }
  if (options.features) {
    exclude.push(options.features);
  }

  let columns =
    options.columns ?? (await columnDescriptions(coordinator, table)).filter((x) => !x.name.startsWith("__"));

  let charts: BuiltinChartSpec[] = [];

  let candidates = columns.filter((item) => {
    if (item.jsType == null) return false;
    if (item.jsType === "string[]") return false;
    if (config.include != undefined && config.include.indexOf(item.name) < 0) return false;
    if (exclude.indexOf(item.name) >= 0) return false;
    if (config.override?.[item.name] !== undefined) return false;
    return true;
  });
  // Race the wide ``APPROX_COUNT_DISTINCT`` batch against a 15 s
  // timeout. On 322 M-row × 30+-col tables the batch can take 60 s+ —
  // long enough that the side panel never populates and the user
  // (correctly) thinks the app is broken. On timeout we fall back to
  // an empty distinct map: number/Date columns get histograms (which
  // don't need cardinality), string columns get count-plots clamped
  // by the existing distinct-count guard, and the panel appears even
  // if a few low-cardinality string columns get skipped.
  let distinctMap: Map<string, number>;
  try {
    distinctMap = await Promise.race([
      distinctCountBatch(
        coordinator,
        table,
        candidates.map((c) => c.name),
      ),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("distinctCountBatch timeout (15 s) — falling back to heuristics")), 15_000),
      ),
    ]);
  } catch (err) {
    console.warn(`[atlas-charts] ${(err as Error).message}`);
    distinctMap = new Map();
  }

  for (let item of columns) {
    if (item.jsType == null) {
      continue;
    }

    // If include is specified, only process columns in the include list.
    if (config.include != undefined && config.include.indexOf(item.name) < 0) {
      continue;
    }
    // If exclude is specified, skip excluded columns.
    if (exclude.indexOf(item.name) >= 0) {
      continue;
    }

    // If we have an override, use the override directly.
    let override = config.override?.[item.name];
    if (override !== undefined) {
      if (override !== null) {
        charts.push(override);
      }
      continue;
    }

    let distinctRaw = distinctMap.get(item.name);
    // ``distinct`` UNKNOWN (timeout fallback) is meaningfully different
    // from "distinct == 2" — for unknown we should be conservative.
    // Default unknowns to a high value so we never build a count-plot
    // for a potentially-high-cardinality string column (which would
    // try to render thousands of bars and freeze layout).
    let distinct = distinctRaw ?? Number.POSITIVE_INFINITY;
    let distinctKnown = distinctRaw !== undefined;
    // Skip the column if there's only a single unique value (only
    // when we actually measured it).
    if (distinctKnown && distinct <= 1) {
      continue;
    }

    switch (item.jsType) {
      case "string": {
        // count-plot is only safe when we KNOW the cardinality is
        // bounded. On the timeout fallback path we skip strings
        // entirely rather than guess — better an empty entry than a
        // 322 M-row count-plot that hangs the renderer.
        if (distinctKnown && distinct <= 1000) {
          charts.push({
            type: "count-plot",
            title: item.name,
            data: { field: item.name },
          });
        }
        break;
      }
      case "string[]": {
        charts.push({
          type: "count-plot",
          title: item.name,
          data: { field: item.name, isList: true },
        });
        break;
      }
      case "number":
      case "Date": {
        // Treat floating-point columns as continuous regardless of cardinality, so a
        // few fractional values produce a value-ordered histogram instead of a
        // frequency-sorted count plot. Low-cardinality integers stay categorical.
        if (distinctKnown && distinct <= 10 && !isFloatingPointDBType(item.type)) {
          charts.push({
            type: "count-plot",
            title: item.name,
            data: { field: item.name },
          });
        } else {
          charts.push(histogramSpec(item.name));
        }
        break;
      }
    }
  }
  return charts;
}

/** Backwards-compatible single-call API. Prefer the two-phase
 *  ``defaultPrimaryCharts`` + ``defaultColumnCharts`` for interactive UIs
 *  so the embedding scatter can race the column distinct-count discovery. */
export async function defaultCharts(options: {
  coordinator: Coordinator;
  table: string;
  id: string;
  projection?: ProjectionInput;
  features?: string;
  config?: DefaultChartsConfig;
}): Promise<BuiltinChartSpec[]> {
  const primary = defaultPrimaryCharts({
    projection: options.projection,
    features: options.features,
    config: options.config,
  });
  const cols = await defaultColumnCharts({
    coordinator: options.coordinator,
    table: options.table,
    projection: options.projection,
    features: options.features,
    config: options.config,
  });
  return [...primary, ...cols];
}

export function histogramSpec(field: string, groupField?: string, table?: string): ChartSpec {
  return {
    title: field,
    layers: [
      {
        ...(table != null ? { from: table } : {}),
        mark: "bar",
        style: { fillColor: "$markColorFade" },
        encoding: {
          x: { field: field },
          y: { aggregate: "count" },
        },
      },
      {
        ...(table != null ? { from: table } : {}),
        mark: "bar",
        filter: "$filter",
        encoding: {
          x: { field: field },
          y: { aggregate: "count" },
          ...(groupField ? { color: { field: groupField } } : {}),
        },
      },
    ],
    selection: { brush: { encoding: "x" } },
    widgets: [
      { type: "scale.type", channel: "x" },
      { type: "encoding.normalize", attribute: "y", layer: [0, 1], options: ["x"] },
    ],
  };
}
