// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Component } from "svelte";

import BoxPlot from "./basic/BoxPlot.svelte";
import ContentViewer from "./basic/ContentViewer.svelte";
import CountPlot from "./basic/CountPlot.svelte";
import CountPlotList from "./basic/CountPlotList.svelte";
import Histogram from "./basic/Histogram.svelte";
import Histogram2D from "./basic/Histogram2D.svelte";
import HistogramStack from "./basic/HistogramStack.svelte";
import Markdown from "./basic/Markdown.svelte";
import Placeholder from "./basic/Placeholder.svelte";
import Predicates from "./basic/Predicates.svelte";
import Builder from "./builder/Builder.svelte";
import Embedding from "./embedding/Embedding.svelte";
import Table from "./table/Table.svelte";

import type {
  BoxPlotSpec,
  ContentViewerSpec,
  CountPlotSpec,
  Histogram2DSpec,
  HistogramSpec,
  HistogramStackSpec,
  MarkdownSpec,
  PredicatesSpec,
} from "./basic/types.js";
import type { UIElement } from "./builder/builder_description.js";
import type { ChartBuilderDescription, ChartViewProps } from "./chart.js";
import type { EmbeddingSpec } from "./embedding/types.js";
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
  return Placeholder;
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
registerChartType("histogram", Histogram);
registerChartType("histogram-stack", HistogramStack);
registerChartType("histogram-2d", Histogram2D);
registerChartType("box-plot", BoxPlot);
registerChartType("embedding", Embedding);
registerChartType("predicates", Predicates);
registerChartType("table", Table);
registerChartType("markdown", Markdown, { supportsEditMode: true });
registerChartType("content-viewer", ContentViewer);

// Spec type for all builtin chart types
export type BuiltinChartSpec =
  | BoxPlotSpec
  | HistogramSpec
  | Histogram2DSpec
  | HistogramStackSpec
  | CountPlotSpec
  | PredicatesSpec
  | EmbeddingSpec
  | TableSpec
  | MarkdownSpec
  | ContentViewerSpec;

// Chart builders

registerChartBuilder({
  icon: "chart-h-bar",
  description: "Create a count plot of a categorical field",
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
  icon: "chart-v-histogram",
  description: "Create a histogram of a field",
  ui: [
    { field: { key: "x", label: "Field", types: ["number", "string"], required: true } }, //
  ] as const,
  create: ({ x }): HistogramSpec | undefined => ({
    type: "histogram",
    title: x.name,
    data: { field: x.name },
    binCount: 20,
  }),
});

registerChartBuilder({
  icon: "chart-stacked",
  description: "Create a stacked histogram",
  ui: [
    { field: { key: "x", label: "X Field", types: ["number", "string"], required: true } }, //
    { field: { key: "y", label: "Group Field", types: ["number", "string"] } }, //
  ] as const,
  create: ({ x, y }): HistogramSpec | HistogramStackSpec | undefined => {
    if (y == null) {
      return {
        type: "histogram",
        title: `${x.name}`,
        data: {
          field: x.name,
        },
        binCount: 20,
      };
    } else {
      return {
        type: "histogram-stack",
        title: `${x.name} by ${y.name}`,
        data: {
          x: x.name,
          group: y.name,
        },
        xBinCount: 20,
        groupBinCount: 5,
      };
    }
  },
});

registerChartBuilder({
  icon: "chart-heatmap",
  description: "Create a 2D heatmap of two fields",
  ui: [
    { field: { key: "x", label: "X Field", types: ["number", "string"], required: true } }, //
    { field: { key: "y", label: "Y Field", types: ["number", "string"], required: true } }, //
  ] as const,
  create: ({ x, y }): Histogram2DSpec | undefined => ({
    type: "histogram-2d",
    title: `${x.name}, ${y.name}`,
    data: { x: x.name, y: y.name },
    xBinCount: 20,
    yBinCount: 20,
  }),
});

registerChartBuilder({
  icon: "chart-boxplot",
  description: "Create a box plot",
  ui: [
    { field: { key: "x", label: "X Field", required: true } }, //
    { field: { key: "y", label: "Y Field", types: ["number"], required: true } }, //
  ] as const,
  create: ({ x, y }): BoxPlotSpec | undefined => ({
    type: "box-plot",
    title: `${y.name} by ${x.name}`,
    data: { x: x.name, y: y.name },
    xBinCount: 20,
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
  ui: [{ code: { key: "spec", language: "json" } }] as const,
  create: ({ spec }): any | undefined => JSON.parse(spec),
});
