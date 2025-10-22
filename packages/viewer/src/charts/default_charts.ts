// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";

import { columnDescriptions, distinctCount } from "../utils/database.js";
import type { BuiltinChartSpec } from "./chart_types.js";

/** Returns a list of default charts for a given data table. */
export async function defaultCharts(
  coordinator: Coordinator,
  table: string,
  id: string,
  options: {
    exclude?: string[];
    projection?: { x: string; y: string; text?: string };
  } = {},
): Promise<BuiltinChartSpec[]> {
  let exclude = options.exclude ?? [];

  let columns = (await columnDescriptions(coordinator, table)).filter((x) => !x.name.startsWith("__"));

  let charts: BuiltinChartSpec[] = [];

  if (options?.projection != null) {
    charts.push({
      type: "embedding",
      title: "Embedding",
      data: {
        x: options.projection.x,
        y: options.projection.y,
        text: options.projection.text,
      },
    });
  }

  charts.push({ type: "predicates", title: "SQL Predicates" });
  charts.push({ type: "table", title: "Table", columns: columns.map((x) => x.name) });

  for (let item of columns) {
    if (item.jsType == null) {
      continue;
    }
    if (exclude.indexOf(item.name) >= 0) {
      continue;
    }
    let distinct = await distinctCount(coordinator, table, item.name);
    // Skip the column if there's only a single unique value.
    if (distinct <= 1) {
      continue;
    }
    switch (item.jsType) {
      case "string":
        if (distinct <= 1000) {
          charts.push({
            type: "count-plot",
            title: item.name,
            data: { field: item.name },
          });
        }
        break;
      case "string[]":
        charts.push({
          type: "count-plot-list",
          title: item.name,
          data: { field: item.name },
        });
        break;
      case "number":
        if (distinct <= 10) {
          charts.push({
            type: "count-plot",
            title: item.name,
            data: { field: item.name },
          });
        } else {
          charts.push({
            type: "histogram",
            title: item.name,
            data: { field: item.name },
            binCount: 20,
          });
        }
        break;
    }
  }
  return charts;
}
