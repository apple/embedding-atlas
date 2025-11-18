// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Component } from "svelte";

import ContentViewer from "./basic/ContentViewer.svelte";
import CountPlot from "./basic/CountPlot.svelte";
import CountPlotList from "./basic/CountPlotList.svelte";
import Markdown from "./basic/Markdown.svelte";
import Placeholder from "./basic/Placeholder.svelte";
import Predicates from "./basic/Predicates.svelte";
import Builder from "./builder/Builder.svelte";
import Embedding from "./embedding/Embedding.svelte";
import Chart from "./spec/Chart.svelte";
import Table from "./table/Table.svelte";

import type { ContentViewerSpec, CountPlotSpec, MarkdownSpec, PredicatesSpec } from "./basic/types.js";
import type { UIElement } from "./builder/builder_description.js";
import type { ChartBuilderDescription, ChartViewProps } from "./chart.js";
import { histogramSpec } from "./default_charts.js";
import type { EmbeddingSpec } from "./embedding/types.js";
import type { ChartSpec } from "./spec/spec.js";
import type { TableSpec } from "./table/types.js";

export type ChartComponent = Component<ChartViewProps<any, any>, {}, "">;

interface ChartTypeOptions {
  /**
   * The chart component supports edit mode.
   * If set to true, the chart component is responsible for editing the chart.
   * Otherwise, a JSON spec editor will be used.
   */
  supportsEditMode?: boolean;
}

const chartTypes: Record<string, ChartComponent> = {};
const chartTypeOptions: Record<string, ChartTypeOptions> = {};
const chartBuilders: ChartBuilderDescription<any, any>[] = [];

export function registerChartType(type: string, component: ChartComponent, options: ChartTypeOptions = {}) {
  chartTypes[type] = component;
  chartTypeOptions[type] = options;
}

export function registerChartBuilder<Spec, T extends readonly UIElement[]>(builder: ChartBuilderDescription<Spec, T>) {
  chartBuilders.push(builder);
}

export function findChartComponent(spec: any): ChartComponent {
  if (typeof spec != "object") {
    return Placeholder;
  }
  if (typeof spec.type == "string") {
    let r = chartTypes[spec.type];
    if (r == null) {
      return Placeholder;
    }
    return r;
  }
  return Chart;
}

export function findChartTypeOptions(spec: any): ChartTypeOptions {
  if (typeof spec != "object") {
    return {};
  }
  if (typeof spec.type == "string") {
    let r = chartTypeOptions[spec.type];
    if (r == null) {
      return {};
    }
    return r;
  }
  return {};
}

export function chartBuilderDescriptions(): ChartBuilderDescription<any, any>[] {
  return chartBuilders;
}

// Chart builder is a special chart type.
registerChartType("builder", Builder);

// Builtin chart types
registerChartType("count-plot", CountPlot);
registerChartType("count-plot-list", CountPlotList);
registerChartType("embedding", Embedding);
registerChartType("predicates", Predicates);
registerChartType("table", Table);
registerChartType("markdown", Markdown, { supportsEditMode: true });
registerChartType("content-viewer", ContentViewer);

// Spec type for all builtin chart types
export type BuiltinChartSpec =
  | ChartSpec
  | ContentViewerSpec
  | CountPlotSpec
  | EmbeddingSpec
  | MarkdownSpec
  | PredicatesSpec
  | TableSpec;

// Chart builders

registerChartBuilder({
  icon: "chart-h-bar",
  description: "Create a count plot of a field",
  ui: [
    { field: { key: "x", label: "Field", types: ["number", "string", "string[]"], required: true } }, //
  ] as const,
  create: ({ x }): CountPlotSpec | undefined => {
    if (x.type == "discrete[]") {
      return {
        title: x.name,
        type: "count-plot-list",
        data: { field: x.name },
      };
    } else {
      return {
        title: x.name,
        type: "count-plot",
        data: { field: x.name },
      };
    }
  },
});

registerChartBuilder({
  icon: "chart-stacked",
  description: "Create a histogram of a field",
  ui: [
    { field: { key: "x", label: "Field", types: ["number", "string"], required: true } }, //
    { field: { key: "color", label: "Group Field", types: ["number", "string"] } },
  ] as const,
  create: ({ x, color }): ChartSpec | undefined => histogramSpec(x.name, color?.name),
});

registerChartBuilder({
  icon: "chart-line",
  description: "Create a histogram of a field",
  ui: [
    { field: { key: "x", label: "X Field", types: ["number", "string"], required: true } }, //
    { field: { key: "y", label: "Y Field", types: ["number"], required: true } }, //
    { field: { key: "color", label: "Group Field", types: ["number", "string"] } },
  ] as const,
  create: ({ x, y, color }): ChartSpec | undefined => ({
    title: y.name,
    layers: [
      {
        mark: "line",
        filter: "$filter",
        encoding: {
          x: { field: x.name },
          y: { aggregate: "mean", field: y.name },
          ...(color ? { color: { field: color.name } } : {}),
        },
      },
    ],
    selection: { brush: { encoding: "x" } },
    widgets: [
      { type: "scale.type", channel: "x" },
      { type: "encoding.normalize", attribute: "y", layer: 0, options: ["x"] },
    ],
  }),
});

registerChartBuilder({
  icon: "chart-ecdf",
  description: "Create a chart showing the empirical cumulative distribution (eCDF) of a field",
  ui: [
    { field: { key: "x", label: "Field", types: ["number"], required: true } }, //
    { field: { key: "color", label: "Group Field", types: ["number", "string"] } },
  ] as const,
  create: ({ x, color }): ChartSpec | undefined => ({
    title: x.name,
    layers: [
      {
        mark: "line",
        filter: "$filter",
        encoding: {
          x: { aggregate: "ecdf-value", field: x.name },
          y: { aggregate: "ecdf-rank" },
          ...(color ? { color: { field: color.name } } : {}),
        },
      },
    ],
    selection: { brush: { encoding: "x" } },
    widgets: [{ type: "scale.type", channel: "x" }],
  }),
});

registerChartBuilder({
  icon: "chart-heatmap",
  description: "Create a 2D heatmap of two fields",
  ui: [
    { field: { key: "x", label: "X Field", types: ["number", "string"], required: true } }, //
    { field: { key: "y", label: "Y Field", types: ["number", "string"], required: true } }, //
  ] as const,
  create: ({ x, y }): ChartSpec | undefined => ({
    title: `${x.name}, ${y.name}`,
    layers: [
      {
        mark: "rect",
        filter: "$filter",
        zIndex: -1,
        encoding: {
          x: { field: x.name },
          y: { field: y.name },
          color: { aggregate: "count" },
        },
      },
      {
        mark: "rect",
        zIndex: -2,
        encoding: {
          color: {
            value: 0,
          },
        },
      },
    ],
    selection: { brush: { encoding: "xy" } },
    widgets: [
      { type: "scale.type", channel: "x" },
      { type: "scale.type", channel: "y" },
      { type: "encoding.normalize", attribute: "color", layer: 0, options: ["x", "y"] },
    ],
  }),
});

registerChartBuilder({
  icon: "chart-boxplot",
  description: "Create a box plot",
  ui: [
    { field: { key: "x", label: "X Field", required: true } }, //
    { field: { key: "y", label: "Y Field", types: ["number"], required: true } }, //
  ] as const,
  create: ({ x, y }): ChartSpec | undefined => ({
    title: x.name,
    layers: [
      {
        mark: "rect",
        filter: "$filter",
        width: 1,
        style: { fillColor: "$ruleColor" },
        encoding: {
          x: { field: x.name },
          y1: { aggregate: "min", field: y.name },
          y2: { aggregate: "max", field: y.name },
        },
      },
      {
        mark: "rect",
        filter: "$filter",
        width: { gap: 1, clampToRatio: 0.1 },
        encoding: {
          x: { field: x.name },
          y1: { aggregate: "quantile", quantile: 0.25, field: y.name },
          y2: { aggregate: "quantile", quantile: 0.75, field: y.name },
        },
      },
      {
        mark: "rect",
        filter: "$filter",
        height: 1,
        width: { gap: 1, clampToRatio: 0.1 },
        style: { fillColor: "$ruleColor" },
        encoding: {
          x: { field: x.name },
          y: { aggregate: "median", field: y.name },
        },
      },
    ],
    selection: { brush: { encoding: "x" } },
    axis: {
      y: { title: y.name },
    },
    widgets: [
      { type: "scale.type", channel: "x" },
      { type: "scale.type", channel: "y" },
    ],
  }),
});

registerChartBuilder({
  icon: "chart-bubble",
  description: "Create a bubble chart",
  ui: [
    { field: { key: "x", label: "X Field", types: ["number"], required: true } }, //
    { field: { key: "y", label: "Y Field", types: ["number"], required: true } }, //
    { field: { key: "color", label: "Color Field", types: ["number", "string"] } }, //
    { field: { key: "group", label: "Group Field", types: ["number", "string"] } }, //
  ] as const,
  create: ({ x, y, color, group }): ChartSpec | undefined => ({
    title: x.name,
    layers: [
      {
        mark: "point",
        filter: "$filter",
        style: {
          fillColor: "$encoding",
          fillOpacity: 0.1,
          strokeColor: "$encoding",
        },
        encoding: {
          x: { aggregate: "mean", field: x.name },
          y: { aggregate: "mean", field: y.name },
          size: { aggregate: "count" },
          ...(color ? { color: { field: color?.name } } : {}),
          ...(group ? { group: { field: group.name } } : {}),
        },
      },
    ],
    selection: { brush: { encoding: "xy" } },
    widgets: [
      { type: "scale.type", channel: "x" },
      { type: "scale.type", channel: "y" },
    ],
  }),
});

registerChartBuilder({
  icon: "chart-embedding",
  description: "Create an embedding view",
  ui: [
    { field: { key: "x", label: "X Field", types: ["number"], required: true } }, //
    { field: { key: "y", label: "Y Field", types: ["number"], required: true } }, //
    { field: { key: "text", label: "Text Field", types: ["string"] } }, //
    { field: { key: "category", label: "Category Field", types: ["string", "number"] } }, //
  ] as const,
  preview: false,
  create: ({ x, y, text, category }, context): EmbeddingSpec | undefined => ({
    type: "embedding",
    title: "Embedding",
    data: {
      x: x.name,
      y: y.name,
      text: text?.name,
      category: category?.name,
    },
  }),
});

registerChartBuilder({
  icon: "chart-predicates",
  description: "Create a filter with custom SQL predicates",
  ui: [] as const,
  create: (): PredicatesSpec | undefined => ({
    type: "predicates",
    title: "SQL Predicates",
  }),
});

registerChartBuilder({
  icon: "chart-markdown",
  description: "Create a view with markdown content",
  preview: false,
  ui: [{ code: { key: "content", language: "markdown" } }] as const,
  create: ({ content }): any | undefined => ({
    type: "markdown",
    title: "Markdown",
    content: content,
  }),
});

registerChartBuilder({
  icon: "chart-content-viewer",
  description: "Create a view that displays a given field's content for the last selected point",
  preview: false,
  ui: [{ field: { key: "field", label: "Field", required: true } }] as const,
  create: ({ field }): ContentViewerSpec | undefined => ({
    type: "content-viewer",
    title: field.name,
    field: field.name,
  }),
});

registerChartBuilder({
  icon: "chart-spec",
  description: "Create a chart with custom spec",
  preview: false,
  ui: [{ spec: { key: "spec" } }] as const,
  create: ({ spec }): ChartSpec | undefined => spec,
});
