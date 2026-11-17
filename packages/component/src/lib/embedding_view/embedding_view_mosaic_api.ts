// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator, Selection } from "@uwdata/mosaic-core";
import { createClassComponent } from "svelte/legacy";

import Component from "./EmbeddingViewMosaic.svelte";

import type { Point, Rectangle, ViewportState } from "../utils.js";
import type { EmbeddingViewConfig } from "./embedding_view_config.js";
import type { ColorScale, SQLField } from "./data/fields.js";
import type { ThemeConfig } from "./theme.js";
import type { Cache, CustomComponent, DataField, DataPoint, DataPointID, Label, OverlayProxy } from "./types.js";

/** The data of a layer: a table and its columns. Fields can be column names or SQL expressions (`{ sql: "..." }`,
 *  evaluated for the layer's table). */
interface LayerData {
  /** The data table. Defaults to the component's `table`.
   *  If this is a different table, the fields below are not inherited from the component. */
  table?: string;
  /** The table's identifier (row id) column, which the table's data is loaded by (and which image labels are
   *  looked up by). Defaults to the component's `identifier` on the component's table, otherwise DuckDB's `rowid`
   *  (which views don't have: specify an identifier for layers on views). */
  identifier?: string | null;
  /** The x field. Defaults to the component's `x`. */
  x?: SQLField;
  /** The y field. Defaults to the component's `y`. */
  y?: SQLField;
  /** The category field (points and density layers). Defaults to the component's `category` (unless the layer
   *  specifies a `color`). */
  category?: SQLField | null;
  /** The text field for automatic labels (labels layers). Defaults to the component's `text`. */
  text?: SQLField | null;
  /** The image field for image labels (labels layers). Defaults to the component's `image`. */
  image?: SQLField | null;
  /** The importance field for image labels (labels layers). Defaults to the component's `importance`. */
  importance?: SQLField | null;
}

/** A layer that draws the data as points. */
interface PointsLayer {
  type: "points";
  /** A stable identifier for the layer. Defaults to the layer's index. */
  id?: string;
  /** Whether the layer is visible. Default true. */
  visible?: boolean;
  /** The layer's data. Fields not specified are inherited from the component (on the component's table). */
  data?: LayerData;
  /** A Mosaic `Selection` object to filter the layer.
   *  Defaults to the component's `filter` if the layer uses the component's table, otherwise no filter
   *  (the component's `filter` applies to the component's table). Set to `null` to show all data. */
  filter?: Selection | null;
  /** A fixed color for the layer, used when the layer has no category. A layer with a `color` doesn't inherit the
   *  component's category. Defaults to the component's `color`. */
  color?: string | null;
  /** How the layer's categories map to colors (see `ColorScale`). Defaults to the component's `colorScale` if the
   *  layer uses the component's category, otherwise to default colors for the layer's categories (as indices). */
  colorScale?: ColorScale | null;
  /** The opacity of the layer, a number between 0 and 1, or "auto" (default).
   *  With "auto", the opacity increases as you zoom in. */
  opacity?: number | "auto";
  /** The point size. Defaults to `config.pointSize`, or automatically calculated based on density. */
  pointSize?: number | null;
  /** Approximate maximum number of points to render. Defaults to `config.downsampleMaxPoints`. */
  downsampleMaxPoints?: number | null;
  /** Density weight for downsampling (0-10). Defaults to `config.downsampleDensityWeight`. */
  downsampleDensityWeight?: number | null;
}

/** A layer that draws the density of the data as a filled density map and contours. */
interface DensityLayer {
  type: "density";
  /** A stable identifier for the layer. Defaults to the layer's index. */
  id?: string;
  /** Whether the layer is visible. Default true. */
  visible?: boolean;
  /** The layer's data. Fields not specified are inherited from the component (on the component's table). */
  data?: LayerData;
  /** A Mosaic `Selection` object to filter the layer.
   *  Defaults to the component's `filter` if the layer uses the component's table, otherwise no filter
   *  (the component's `filter` applies to the component's table). Set to `null` to show all data. */
  filter?: Selection | null;
  /** A fixed color for the layer, used when the layer has no category. A layer with a `color` doesn't inherit the
   *  component's category. Defaults to the component's `color`. */
  color?: string | null;
  /** How the layer's categories map to colors (see `ColorScale`). Defaults to the component's `colorScale` if the
   *  layer uses the component's category, otherwise to default colors for the layer's categories (as indices). */
  colorScale?: ColorScale | null;
  /** The opacity of the layer, a number between 0 and 1, or "auto" (default).
   *  With "auto", the density fades out as you zoom in. */
  opacity?: number | "auto";
  /** Whether to draw the filled density map. Default true. */
  fill?: boolean | null;
  /** Whether to draw the density contours. Default true. */
  contours?: boolean | null;
}

/** A layer that draws labels, either supplied directly or generated automatically from the data. */
interface LabelsLayer {
  type: "labels";
  /** A stable identifier for the layer. Defaults to the layer's index. */
  id?: string;
  /** Whether the layer is visible. Default true. */
  visible?: boolean;
  /** The opacity of the labels, a number between 0 and 1. Defaults to the theme's `clusterLabelOpacity`. */
  opacity?: number | null;
  /** The labels to display. If not specified, labels are generated automatically
   *  from the layer's data (using the `text` field, or the `image` and `importance` fields). */
  labels?: Label[] | null;
  /** The layer's data (`table`, `identifier`, `x`, `y`, `text`, `image`, `importance`). Fields not specified are
   *  inherited from the component (on the component's table). */
  data?: Omit<LayerData, "category">;
  /** A Mosaic `Selection` object to filter the data for automatic labels.
   *  If specified, labels are regenerated when the filter changes.
   *  Defaults to `null`: labels are generated once from all data (unlike points and density layers,
   *  this does not default to the component's `filter`). */
  filter?: Selection | null;
  /** The density threshold to filter the clusters before generating automatic labels.
   *  Defaults to `config.autoLabelDensityThreshold`. */
  densityThreshold?: number | null;
  /** The stop words for automatic label generation. Defaults to `config.autoLabelStopWords`. */
  stopWords?: string[] | null;
}

export interface EmbeddingViewMosaicProps {
  /** The Mosaic coordinator.
   *  If not specified, the default coordinator from Mosaic's `coordinator()` method will be used. */
  coordinator?: Coordinator;

  /** The data table name. The view loads the table's columns once (by `identifier`, or DuckDB's `rowid`; a view
   *  needs an `identifier`), and filters them locally. If the table's rows change, use a new view (or coordinator)
   *  to reload them. */
  table: string;

  /** The x field: a column name, or a SQL expression (`{ sql: "..." }`). */
  x: SQLField;

  /** The y field: a column name, or a SQL expression (`{ sql: "..." }`). */
  y: SQLField;

  /** The category field (a column name, or a SQL expression), which colors the points and splits the density.
   *  How its values map to categories and colors is set by `colorScale`. Without a `colorScale` domain (or
   *  thresholds), the values must be category indices: integers starting from 0. */
  category?: SQLField | null;

  /** How the category values map to colors: a `domain` of values (compared as text) or numeric `thresholds`, and
   *  the colors (`range`, and the `other` and `null` colors). See `ColorScale`. */
  colorScale?: ColorScale | null;

  /** A fixed color for the points and density when there's no `category`. */
  color?: string | null;

  /** The text field (a column name, or a SQL expression).
   *  If specified, the default tooltip shows the text content.
   *  The text content is also used to generate labels automatically. */
  text?: SQLField | null;

  /** The image field (a column name, or a SQL expression).
   *  If specified along with `importance`, cluster labels will display the highest-importance image per region. */
  image?: SQLField | null;

  /** The importance score field (e.g., PageRank, centrality), a column name or a SQL expression.
   *  Used together with `image` to select representative images for cluster labels. */
  importance?: SQLField | null;

  /** The name of the identifier (aka., id) column: unique per row (e.g., a row index).
   *  If specified, the `selection` object will contain an `identifier` property that you can use to identify the point.
   *  The table's data is loaded and filtered by it; without it, by DuckDB's `rowid` (which views don't have). */
  identifier?: string | null;

  /** Additional fields for the tooltip data element.
   *  Each field can be specified as a column name or a SQL expression. */
  additionalFields?: Record<string, DataField> | null;

  /** The colors for the categories (category indices). Category `i` will use the `i`-th color from this list.
   *  Ignored if `colorScale` is specified (use `colorScale.range`). If not specified, default colors will be used. */
  categoryColors?: string[] | null;

  /** A Mosaic `Selection` object to filter the contents of this view. */
  filter?: Selection | null;

  /** Labels to display on the embedding view.
   *  Each label must have `x`, `y`, and `content` properties,
   *  and optionally `level` and `priority`.
   *  Ignored if `layers` is specified (use a labels layer instead). */
  labels?: Label[] | null;

  /** The layers to draw, which can be points, density, and labels layers.
   *  Each layer can have its own data (table and fields), filter, and colors, defaulting to the component's.
   *  Points and density layers are blended together regardless of their order;
   *  labels are drawn on top.
   *  Tooltip and selection interactions use the component's main data (`table`, `x`, `y`, and `filter`).
   *  If not specified, the layers are determined by `config.mode` and `labels`:
   *  a points layer (with a density layer in density mode), and a labels layer. */
  layers?: (PointsLayer | DensityLayer | LabelsLayer)[] | null;

  /** The width of the view. */
  width?: number | null;

  /** The height of the view. */
  height?: number | null;

  /** The pixel ratio of the view. */
  pixelRatio?: number | null;

  /** Configure the theme of the view. */
  theme?: ThemeConfig | null;

  /** Configure the embedding view. */
  config?: EmbeddingViewConfig | null;

  /** The viewport state.
   *  You may use this to share viewport state across multiple views.
   *  If undefined or set to `null`, the view will use a default viewport state.
   *  To listen to viewport state change, use `onViewportState`. */
  viewportState?: ViewportState | null;

  /** The current tooltip.
   *  The tooltip is an object with the following fields: `x`, `y`, `category`,
   *  `text`, `identifier`.
   *
   *  You may pass the identifier for the data point (`DataPointID`), or a `DataPoint`
   *  object, or a Mosaic `Selection`. If an id or a `DataPoint` object is specified,
   *  you will need to use `onTooltip` to listen to tooltip changes; if a Mosaic
   *  `Selection` is used, the selection will be updated when tooltip is triggered.
   */
  tooltip?: Selection | DataPoint | DataPointID | null;

  /** The current single or multiple point selection.
   *
   *  You may pass an array of `DataPointID` or `DataPoint` objects, or a Mosaic
   *  `Selection`. If `DataPointID[]` or `DataPoint[]` is specified, you will need
   *  to use `onSelection` to listen to selection changes; if a Mosaic `Selection`
   *  is used, the selection will be updated with the appropriate predicates. */
  selection?: Selection | DataPoint[] | DataPointID[] | null;

  /** A Mosaic `Selection` object to capture the component's range selection. */
  rangeSelection?: Selection | null;

  /** The rectangle or polygon that drives the range selection. Setting this
   *  changes the current range selection and also affects the selection passed
   *  in `rangeSelection`. Use `onRangeSelection` to listen for changes to this
   *  rectangle. */
  rangeSelectionValue?: Rectangle | Point[] | null;

  /** A callback for when `viewportState` changes. */
  onViewportState?: ((value: ViewportState) => void) | null;

  /** A callback for when `tooltip` changes. */
  onTooltip?: ((value: DataPoint | null) => void) | null;

  /** A callback for when `selection` changes. */
  onSelection?: ((value: DataPoint[] | null) => void) | null;

  /** A callback for when `rangeSelection` changes. */
  onRangeSelection?: ((value: Rectangle | Point[] | null) => void) | null;

  /** A callback for the effective values of the layers (read-only), including automatically determined
   *  values (e.g., the opacity with `opacity: "auto"`, and automatically generated labels).
   *  `pointSize` is in CSS pixels. `labels` is null if there are no labels, or they are still being generated.
   *  This is called whenever any value changes, including continuously while zooming.
   *  Unchanged entries (and labels) keep their identity, so you can compare them by reference. */
  onLayerValues?:
    | ((
        values: (
          | { type: "points"; id: string; opacity: number; pointSize: number }
          | { type: "density"; id: string; opacity: number }
          | { type: "labels"; id: string; opacity: number; labels: Label[] | null }
        )[],
      ) => void)
    | null;

  /** A custom renderer to draw the tooltip content. */
  customTooltip?: CustomComponent<HTMLDivElement, { tooltip: DataPoint }> | null;

  /** A custom renderer to draw overlay on top of the embedding view. */
  customOverlay?: CustomComponent<HTMLDivElement, { proxy: OverlayProxy }> | null;

  /** A cache for intermediate results, such as automatic labels (keyed by the table and columns they're
   *  generated from, so use a cache that's specific to the database). */
  cache?: Cache | null;
}

export class EmbeddingViewMosaic {
  private component: any;
  private currentProps: EmbeddingViewMosaicProps;

  constructor(target: HTMLElement, props: EmbeddingViewMosaicProps) {
    this.currentProps = { ...props };
    this.component = createClassComponent({ component: Component, target: target, props: props });
  }

  update(props: Partial<EmbeddingViewMosaicProps>) {
    let updates: Partial<EmbeddingViewMosaicProps> = {};
    for (let key in props) {
      if ((props as any)[key] !== (this.currentProps as any)[key]) {
        (updates as any)[key] = (props as any)[key];
        (this.currentProps as any)[key] = (props as any)[key];
      }
    }
    this.component.$set(updates);
  }

  destroy() {
    this.component.$destroy();
  }
}
