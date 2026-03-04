// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Component } from "svelte";

import ContentViewer from "./basic/ContentViewer.svelte";
import CountPlot from "./basic/CountPlot.svelte";
import Markdown from "./basic/Markdown.svelte";
import Placeholder from "./basic/Placeholder.svelte";
import Predicates from "./basic/Predicates.svelte";
import Builder from "./builder/Builder.svelte";
import Embedding from "./embedding/Embedding.svelte";
import Instances from "./instances/Instances.svelte";
import Chart from "./spec/Chart.svelte";

import type {
  ContentViewerSpec,
  CountPlotSpec,
  CountPlotState,
  MarkdownSpec,
  PredicatesSpec,
  PredicatesState,
} from "./basic/types.js";
import type { UIElement } from "./builder/builder_description.js";
import type { ChartBuilderDescription, ChartViewProps } from "./chart.js";
import { histogramSpec } from "./default_charts.js";
import type { EmbeddingSpec, EmbeddingState } from "./embedding/types.js";
import type { InstancesSpec } from "./instances/types.js";
import type { ChartSpec, ChartState } from "./spec/spec.js";

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
registerChartType("embedding", Embedding);
registerChartType("instances", Instances);
registerChartType("predicates", Predicates);
registerChartType("markdown", Markdown, { supportsEditMode: true });
registerChartType("content-viewer", ContentViewer);

// Spec type for all builtin chart types
export type BuiltinChartSpec =
  | ChartSpec
  | ContentViewerSpec
  | CountPlotSpec
  | EmbeddingSpec
  | InstancesSpec
  | MarkdownSpec
  | PredicatesSpec;

// State type for all builtin chart types
export type BuiltinChartState = ChartState | EmbeddingState | CountPlotState | PredicatesState;

// Chart builders

registerChartBuilder({
  icon: "chart-h-bar",
  description: "Create a count plot of a field",
  ui: [
    { label: "Field", field: { key: "x", types: ["number", "string", "string[]"], required: true } }, //
  ] as const,
  create: ({ x }): CountPlotSpec | undefined => {
    if (x.type == "discrete[]") {
      return {
        title: x.name,
        type: "count-plot",
        data: { field: x.name, isList: true },
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
    { label: "Field", field: { key: "x", types: ["number", "string"], required: true } }, //
    { label: "Group Field", field: { key: "color", types: ["number", "string"] } },
  ] as const,
  create: ({ x, color }): ChartSpec | undefined => histogramSpec(x.name, color?.name),
});

registerChartBuilder({
  icon: "chart-line",
  description: "Create a line chart of two fields",
  ui: [
    { label: "X Field", field: { key: "x", types: ["number", "string"], required: true } }, //
    { label: "Y Field", field: { key: "y", types: ["number"], required: true } }, //
    { label: "Group Field", field: { key: "color", types: ["number", "string"] } },
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
    { label: "Field", field: { key: "x", types: ["number"], required: true } }, //
    { label: "Group Field", field: { key: "color", types: ["number", "string"] } },
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
    { label: "X Field", field: { key: "x", types: ["number", "string"], required: true } }, //
    { label: "Y Field", field: { key: "y", types: ["number", "string"], required: true } }, //
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
    { label: "X Field", field: { key: "x", required: true } }, //
    { label: "Y Field", field: { key: "y", types: ["number"], required: true } }, //
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
    { label: "X Field", field: { key: "x", types: ["number"], required: true } }, //
    { label: "Y Field", field: { key: "y", types: ["number"], required: true } }, //
    { label: "Color Field", field: { key: "color", types: ["number", "string"] } }, //
    { label: "Group Field", field: { key: "group", types: ["number", "string"] } }, //
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
    { label: "X Field", field: { key: "x", types: ["number"], required: true } }, //
    { label: "Y Field", field: { key: "y", types: ["number"], required: true } }, //
    { label: "Text Field", field: { key: "text", types: ["string"] } }, //
    { label: "Category Field", field: { key: "category", types: ["string", "number"] } }, //
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
  ui: [{ label: "Field", field: { key: "field", required: true } }] as const,
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

registerChartBuilder({
  icon: "chart-table",
  description: "Create a table view with pagination",
  preview: false,
  ui: [
    {
      label: "SQL query for the table (optional)",
      details:
        "Leave empty to show the (filtered) dataset, use $table and $filter to refer to the data table and filter predicate respectively.",
      code: { key: "query", language: "sql" },
    },
  ] as const,
  create: ({ query }): InstancesSpec | undefined => {
    return {
      type: "instances",
      title: "Table",
      viewMode: "table",
      query: query != null && query.trim() != "" ? query : undefined,
    };
  },
});

registerChartBuilder({
  icon: "chart-cards",
  description: "Create a card view with pagination",
  preview: false,
  ui: [
    {
      label: "SQL query for the card view (optional)",
      details:
        "Leave empty to show the (filtered) dataset, use $table and $filter to refer to the data table and filter predicate respectively.",
      code: { key: "query", language: "sql" },
    },
    {
      label: "HTML template for the cards (optional)",
      code: { key: "template", language: "" },
    },
  ] as const,
  create: ({ query, template }): InstancesSpec | undefined => {
    return {
      type: "instances",
      title: "Cards",
      viewMode: "cards",
      query: query != null && query.trim() != "" ? query : undefined,
      cardTemplate: template != null && template.trim() != "" ? template : undefined,
    };
  },
});
