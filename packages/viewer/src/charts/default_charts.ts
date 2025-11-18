// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";

import { columnDescriptions, distinctCount } from "../utils/database.js";
import type { BuiltinChartSpec } from "./chart_types.js";
import type { EmbeddingSpec } from "./embedding/types.js";
import type { ChartSpec } from "./spec/spec.js";
import type { TableSpec } from "./table/types.js";

export interface DefaultChartsConfig {
  /** If specified, only include the given columns */
  include?: string[];

  /** Columns to exclude, applicable if `include` is not specified */
  exclude?: string[];

  /** Override the chart spec for certain columns. If the override is set to `null` the column will be skipped */
  override?: Record<string, BuiltinChartSpec | null>;

  /** Set to false to disable the table, or an object to override spec properties */
  table?: boolean | Partial<TableSpec>;

  /** Set to false to disable the embedding view, or an object to override spec properties */
  embedding?: boolean | Partial<EmbeddingSpec>;
}

/** Returns a list of default charts for a given data table. */
export async function defaultCharts(options: {
  coordinator: Coordinator;
  table: string;
  id: string;
  projection?: { x: string; y: string; text?: string };
  config?: DefaultChartsConfig;
}): Promise<BuiltinChartSpec[]> {
  let { coordinator, table, projection } = options;
  let config = options.config ?? {};
  let exclude = config.exclude ?? [];

  let columns = (await columnDescriptions(coordinator, table)).filter((x) => !x.name.startsWith("__"));

  let charts: BuiltinChartSpec[] = [];

  if (projection != null && config.embedding !== false) {
    let spec: EmbeddingSpec = {
      type: "embedding",
      title: "Embedding",
      data: {
        x: projection.x,
        y: projection.y,
        text: projection.text,
      },
    };
    if (typeof config.embedding == "object") {
      spec = { ...spec, ...config.embedding };
    }
    charts.push(spec);
    exclude.push(projection.x);
    exclude.push(projection.y);
    if (projection.text) {
      exclude.push(projection.text);
    }
  }

  charts.push({ type: "predicates", title: "SQL Predicates" });

  if (config.table !== false) {
    let spec: TableSpec = { type: "table", title: "Table", columns: columns.map((x) => x.name) };
    if (typeof config.table == "object") {
      spec = { ...spec, ...config.table };
    }
    charts.push(spec);
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

    let distinct = await distinctCount(coordinator, table, item.name);
    // Skip the column if there's only a single unique value.
    if (distinct <= 1) {
      continue;
    }

    switch (item.jsType) {
      case "string": {
        if (distinct <= 1000) {
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
          type: "count-plot-list",
          title: item.name,
          data: { field: item.name },
        });
        break;
      }
      case "number": {
        if (distinct <= 10) {
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

export function histogramSpec(field: string, groupField?: string): ChartSpec {
  return {
    title: field,
    layers: [
      {
        mark: "bar",
        style: { fillColor: "$markColorFade" },
        encoding: {
          x: { field: field },
          y: { aggregate: "count" },
        },
      },
      {
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
