// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { EmbeddingViewConfig, Label, Point, Rectangle, ViewportState } from "@embedding-atlas/component";

/** A column name, or a SQL expression (`{ sql: "..." }`, evaluated for the table of the view or layer). */
export type EmbeddingField = string | { sql: string };

/** How the values of the category field map to categories and colors.
 *
 *  String and boolean fields are categories: the values of `domain` (compared as text), or the most frequent values;
 *  other values are "(N others)". Numeric and time fields are binned: values outside the domain (including infinite
 *  values) are "(other)". Null (and NaN) values are "(null)". To color numbers (e.g., cluster ids) as categories,
 *  cast them to text, e.g., `category: { sql: "CAST(cluster AS VARCHAR)" }`.
 *  The colors follow the categories (or bins), in order. */
export interface EmbeddingColorScale {
  /** The scale of the bins of a numeric field: "linear", "log" (zero and negative values are "(other)"), or
   *  "symlog". Defaults to a scale inferred from the data. */
  type?: "linear" | "log" | "symlog";
  /** For categories: the complete list of categories, in order (other values are "other"). Defaults to the most
   *  frequent values. For bins: `[min, max]` (numbers, or date strings for times). Defaults to the data's range. */
  domain?: (string | number | boolean)[];
  /** Binning of numeric and time fields. */
  bin?: {
    /** The desired number of bins (approximate: bins have nice boundaries). Default 5. */
    desiredCount?: number;
  };
  /** For categories: the colors, in order (or an interpolate name, sampled for each category).
   *  For bins: an interpolate name (e.g., "viridis"), or a list of colors to interpolate.
   *  Defaults to the theme's category colors, or its continuous interpolate. */
  range?: string[] | string;
  /** The color of "other" values. Defaults to the theme's `otherColor`. */
  other?: string;
  /** The color of null values. Defaults to the theme's `nullColor`. */
  null?: string;
}

/** The data of a layer. Fields not specified are inherited from `data` (when the layer uses the view's table). */
interface LayerDataSpec {
  /** The data table. Defaults to `data.table`.
   *  If this is a different table, the fields below are not inherited from `data`. */
  table?: string;
  /** The table's row id column. Defaults to `data.id` on the view's table, otherwise the table's id column. */
  id?: string;
  /** The x field. Defaults to `data.x`. */
  x?: EmbeddingField;
  /** The y field. Defaults to `data.y`. */
  y?: EmbeddingField;
  /** The category field of a points or density layer. Defaults to `data.category` (unless the layer has a
   *  `color`). */
  category?: EmbeddingField | null;
  /** The text field for automatic labels. Defaults to `data.text`. */
  text?: EmbeddingField | null;
  /** The image field for image labels. Defaults to `data.image`. */
  image?: EmbeddingField | null;
  /** The importance field for image labels. Defaults to `data.importance`. */
  importance?: EmbeddingField | null;
}

interface DataLayerBase {
  /** A stable identifier for the layer. Defaults to the layer's index. */
  id?: string;
  /** Whether the layer is visible. Default true. */
  visible?: boolean;
  /** The opacity of the layer, a number between 0 and 1, or "auto" (default) for zoom-dependent opacity. */
  opacity?: number | "auto";
  /** The layer's data (table and fields). */
  data?: Omit<LayerDataSpec, "text" | "image" | "importance">;
  /** A fixed color for the layer (e.g., "#1f77b4"), used when the layer has no category. A layer with a color
   *  doesn't inherit `data.category`. Defaults to the spec's `color`. */
  color?: string;
  /** How the layer's categories map to colors. Defaults to the spec's `colorScale` if the layer inherits
   *  `data.category`. */
  colorScale?: EmbeddingColorScale;
  /** "$filter" filters the layer with the filter of its table; null shows all data.
   *  Defaults to "$filter" on the view's table (`data.table`), and to no filter on other tables. */
  filter?: "$filter" | null;
}

/** A layer that draws the data as points. */
interface PointsLayer extends DataLayerBase {
  type: "points";
  /** The point size. Defaults to `pointSize`, or automatically calculated based on density. */
  pointSize?: number;
}

/** A layer that draws the density of the data as a filled density map and contours. */
interface DensityLayer extends DataLayerBase {
  type: "density";
  /** Whether to draw the filled density map. Default true. */
  fill?: boolean;
  /** Whether to draw the density contours. Default true. */
  contours?: boolean;
}

/** A layer that draws labels, either supplied directly or generated automatically from the data. */
interface LabelsLayer {
  type: "labels";
  /** A stable identifier for the layer. Defaults to the layer's index. */
  id?: string;
  /** Whether the layer is visible. Default true. */
  visible?: boolean;
  /** The opacity of the labels, a number between 0 and 1. */
  opacity?: number;
  /** The labels to display: a list of labels (each with `x`, `y`, `content`, and optionally `level` and
   *  `priority`), or "$labels" for the labels provided to the viewer (`embeddingViewLabels`, e.g., the CLI's
   *  `--labels` option). If omitted, labels are generated automatically from the layer's data. */
  labels?: Label[] | "$labels";
  /** The layer's data (table, x, y, and the text, or image and importance, for automatic labels). */
  data?: Omit<LayerDataSpec, "category">;
  /** "$filter" filters the labels' data with the filter of the layer's table, and regenerates the labels
   *  when the filter changes. By default (or null), labels are generated once from all data. */
  filter?: "$filter" | null;
}

export interface EmbeddingSpec {
  type: "embedding";
  title?: string;

  /** The data. Fields are column names or SQL expressions (`{ sql: "..." }`), except `id`. */
  data: {
    /** The data table, leave undefined for the default table. */
    table?: string;

    /** The row id column, must be set when using a non-default table. */
    id?: string;

    x: EmbeddingField;
    y: EmbeddingField;
    text?: EmbeddingField | null;
    image?: EmbeddingField | null;
    importance?: EmbeddingField | null;
    /** The category field, which colors the points and splits the density (see `colorScale`). */
    category?: EmbeddingField | null;
    neighbors?: EmbeddingField | null;
  };

  /** How the category values map to colors. */
  colorScale?: EmbeddingColorScale;

  /** The color of the points and density when there's no category. Defaults to the theme's `embeddingColor`. */
  color?: string;

  mode?: "points" | "density";
  /** Controls where the view transitions from density to points as you zoom in,
   *  for all layers with `opacity: "auto"`. */
  minimumDensity?: number;
  pointSize?: number;
  /** Maximum number of points to render (for downsampling). Default: 4000000. Set to null to disable. */
  downsampleMaxPoints?: number | null;
  config?: EmbeddingViewConfig;

  /** The layers to draw. If omitted, the layers are determined by `mode` and the viewer's options:
   *  points (with density in density mode), and labels: the labels provided to the viewer ("$labels") on the
   *  main table, otherwise automatic labels unless they are disabled (`embeddingViewConfig.autoLabelEnabled`).
   *  The view's brush always applies to the view's table (`data.table`), and doesn't filter the view's own
   *  layers on that table. Layers on other tables with `filter: "$filter"` follow their table's filter, which
   *  includes the view's brush if the table is related to the main table. */
  layers?: (PointsLayer | DensityLayer | LabelsLayer)[];
}

export interface EmbeddingState {
  /** The viewport state */
  viewport?: ViewportState;
  /** State of the legends, keyed by the legend (its table, category, and color scale). */
  legends?: Record<
    string,
    {
      /** Selected categories */
      selection?: string[];
    }
  >;
  /**
   * State of the brush selection. Can be a rectangle or a list of points for a lasso selection.
   * Coordinates should be in data units.
   */
  brush?: Rectangle | Point[];
}
