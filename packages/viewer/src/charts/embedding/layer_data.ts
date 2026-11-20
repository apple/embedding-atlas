// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { fieldKey, type EmbeddingViewMosaicProps } from "@embedding-atlas/component";

import type { ChartContext } from "../chart.js";
import type { EmbeddingColorScale, EmbeddingField, EmbeddingSpec } from "./types.js";

export type LayerSpec = NonNullable<EmbeddingSpec["layers"]>[number];

/** The effective values of a layer, reported by the embedding view's `onLayerValues`. */
export type LayerValues = Parameters<NonNullable<EmbeddingViewMosaicProps["onLayerValues"]>>[0][number];

/** A layer's data source after applying the defaults from `spec.data`. */
export interface LayerData {
  /** The layer's table. */
  table: string;
  /** Whether the layer uses the view's table (`spec.data.table`). Only then are fields inherited from `spec.data`. */
  isDataTable: boolean;
  /** The layer's row id column (`data.id`), if specified (otherwise the table's). */
  id: string | null;
  x: EmbeddingField | undefined;
  y: EmbeddingField | undefined;
  /** The category field of a points or density layer (null if none, or if the layer has a fixed color). */
  category: EmbeddingField | null;
  /** The color scale of the category. */
  colorScale: EmbeddingColorScale | null;
  /** Whether the category and color scale are the view's (`spec.data.category` and `spec.colorScale` on the view's
   *  table): the layer then shares the view's legend. */
  categoryIsMain: boolean;
  /** The fixed color of a points or density layer without a category (null for the default color). */
  color: string | null;
  /** The text, image, and importance fields of a labels layer. */
  text: EmbeddingField | null;
  image: EmbeddingField | null;
  importance: EmbeddingField | null;
  /** Whether the layer is filtered by the filter of its table (`filterFor(table)`). */
  filtered: boolean;
}

/** A stable key of a field (for comparing fields, and in cache keys). */
export { fieldKey };

/** The viewer's options that determine the labels of an embedding view without explicit `layers`. */
export interface DefaultLabelsOptions {
  /** Whether labels are provided to the viewer (`embeddingViewLabels`, e.g., the CLI's `--labels <file>`). */
  providedLabels: boolean;
  /** Whether automatic labels are enabled (`embeddingViewConfig.autoLabelEnabled`, e.g., `--labels disabled`). */
  autoLabels: boolean;
  /** Whether the view's table is the viewer's main table. The provided labels use the main table's coordinates. */
  isMainTable: boolean;
}

/** The default labels options of a view, from the viewer's options. */
export function defaultLabelsOptions(
  context: Pick<ChartContext, "embeddingViewLabels" | "embeddingViewConfig">,
  isMainTable: boolean,
): DefaultLabelsOptions {
  return {
    providedLabels: context.embeddingViewLabels != null,
    autoLabels: context.embeddingViewConfig?.autoLabelEnabled !== false,
    isMainTable,
  };
}

/** The default labels layer: the provided labels ("$labels") on the main table if there are any,
 *  otherwise automatic labels, or none if automatic labels are disabled. */
export function defaultLabelsLayer(options: DefaultLabelsOptions): LayerSpec | null {
  if (options.providedLabels && options.isMainTable) {
    return { type: "labels", labels: "$labels" };
  }
  if (options.autoLabels) {
    return { type: "labels" };
  }
  return null;
}

/** The layers of a spec without `layers`: points (with density in density mode, and the spec's `pointSize`),
 *  and the default labels layer. */
export function defaultLayers(spec: EmbeddingSpec, options: DefaultLabelsOptions): LayerSpec[] {
  let { pointSize } = spec;
  let labels = defaultLabelsLayer(options);
  return [
    ...(spec.mode == "density" ? [{ type: "density" as const }] : []),
    { type: "points" as const, ...(pointSize != null ? { pointSize } : {}) },
    ...(labels != null ? [labels] : []),
  ];
}

/** Whether a layer is filtered when `filter` is not specified:
 *  points and density layers on the view's table are; labels layers and layers on other tables are not. */
export function isFilteredByDefault(type: LayerSpec["type"], isDataTable: boolean): boolean {
  return type != "labels" && isDataTable;
}

/** Resolve a layer's table, fields, colors, and filter, inheriting from `spec.data` when the layer uses the view's
 *  table. */
export function resolveLayerData(spec: EmbeddingSpec, layer: LayerSpec, dataTable: string): LayerData {
  let data: {
    table?: string;
    id?: string;
    x?: EmbeddingField;
    y?: EmbeddingField;
    category?: EmbeddingField | null;
    text?: EmbeddingField | null;
    image?: EmbeddingField | null;
    importance?: EmbeddingField | null;
  } = layer.data ?? {};
  let table = data.table ?? dataTable;
  let isDataTable = table == dataTable;
  let inherit = <T>(value: T | null | undefined, fallback: T | null | undefined): T | null =>
    value !== undefined ? value : isDataTable ? (fallback ?? null) : null;

  let category: EmbeddingField | null = null;
  let colorScale: EmbeddingColorScale | null = null;
  let categoryIsMain = false;
  let color: string | null = null;
  if (layer.type != "labels") {
    // A layer with a fixed color doesn't inherit the view's category.
    category = inherit(data.category, layer.color == null ? spec.data.category : null);
    let sameAsMain = isDataTable && category != null && fieldKey(category) == fieldKey(spec.data.category);
    colorScale = layer.colorScale ?? (sameAsMain ? (spec.colorScale ?? null) : null);
    categoryIsMain = sameAsMain && layer.colorScale === undefined;
    color = category == null ? (layer.color ?? spec.color ?? null) : null;
  }

  return {
    table,
    isDataTable,
    id: data.id ?? null,
    x: data.x ?? (isDataTable ? spec.data.x : undefined),
    y: data.y ?? (isDataTable ? spec.data.y : undefined),
    category,
    colorScale,
    categoryIsMain,
    color,
    text: layer.type == "labels" ? inherit(data.text, spec.data.text) : null,
    image: layer.type == "labels" ? inherit(data.image, spec.data.image) : null,
    importance: layer.type == "labels" ? inherit(data.importance, spec.data.importance) : null,
    filtered: layer.filter === undefined ? isFilteredByDefault(layer.type, isDataTable) : layer.filter == "$filter",
  };
}
