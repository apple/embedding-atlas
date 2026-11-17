// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createClassComponent } from "svelte/legacy";

import Component from "./EmbeddingView.svelte";

import type { Point, Rectangle, ViewportState } from "../utils.js";
import type { EmbeddingViewConfig } from "./embedding_view_config.js";
import type { ThemeConfig } from "./theme.js";
import type { Cache, CustomComponent, DataPoint, Label, LabelContent, OverlayProxy } from "./types.js";

/** A layer that draws the data as points. */
interface PointsLayer {
  type: "points";
  /** A stable identifier for the layer. Defaults to the layer's index. */
  id?: string;
  /** Whether the layer is visible. Default true. */
  visible?: boolean;
  /** The opacity of the layer, a number between 0 and 1, or "auto" (default).
   *  With "auto", the opacity increases as you zoom in. */
  opacity?: number | "auto";
  /** The data for the layer. Defaults to the component's `data`. */
  data?: {
    x: Float32Array<ArrayBuffer>;
    y: Float32Array<ArrayBuffer>;
    category?: Uint8Array<ArrayBuffer> | null;
  } | null;
  /** The colors for the categories. Defaults to the component's `categoryColors` if the layer uses the component's categories, otherwise to default colors for the layer's categories. */
  categoryColors?: string[] | null;
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
  /** The opacity of the layer, a number between 0 and 1, or "auto" (default).
   *  With "auto", the density fades out as you zoom in. */
  opacity?: number | "auto";
  /** The data for the layer. Defaults to the component's `data`. */
  data?: {
    x: Float32Array<ArrayBuffer>;
    y: Float32Array<ArrayBuffer>;
    category?: Uint8Array<ArrayBuffer> | null;
  } | null;
  /** The colors for the categories. Defaults to the component's `categoryColors` if the layer uses the component's categories, otherwise to default colors for the layer's categories. */
  categoryColors?: string[] | null;
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
   *  from the layer's data with `queryClusterLabels`. */
  labels?: Label[] | null;
  /** The data used to find clusters for automatic labels. Defaults to the component's `data`. */
  data?: {
    x: Float32Array<ArrayBuffer>;
    y: Float32Array<ArrayBuffer>;
  } | null;
  /** An async function that returns labels for a list of clusters.
   *  Defaults to the component's `queryClusterLabels`. */
  queryClusterLabels?: ((clusters: Rectangle[][]) => Promise<(LabelContent | null)[]>) | null;
  /** The density threshold to filter the clusters before generating automatic labels.
   *  Defaults to `config.autoLabelDensityThreshold`. */
  densityThreshold?: number | null;
  /** Identifies the layer's data and labels across sessions (any JSON value, e.g., `"topics-v2"`), to store its
   *  automatic labels in the `cache`. Automatic labels of the component's data with the component's
   *  `queryClusterLabels` are cached without it; labels of a layer's own `data` or own `queryClusterLabels` are
   *  only cached with a `cacheKey`. */
  cacheKey?: any;
}

export interface EmbeddingViewProps {
  /** The data. To update the data, pass new arrays (arrays are not expected to change in place). */
  data: {
    /** An array of X coordinates, must be a `Float32Array`. */
    x: Float32Array<ArrayBuffer>;
    /** An array of Y coordinates, must be a `Float32Array`. */
    y: Float32Array<ArrayBuffer>;
    /** An array of category indices, must be a `Uint8Array`. */
    category?: Uint8Array<ArrayBuffer> | null;
  };

  /** The colors for the categories.
   *  Category `i` will use the `i`-th color from this list.
   *  If not specified, default colors will be used. */
  categoryColors?: string[] | null;

  /** Labels to display on the embedding view.
   *  Each label must have `x`, `y`, and `content` properties,
   *  and optionally `level` and `priority`.
   *  Ignored if `layers` is specified (use a labels layer instead). */
  labels?: Label[] | null;

  /** The layers to draw, which can be points, density, and labels layers.
   *  Each layer can have its own data, defaulting to the component's `data`.
   *  Points and density layers are blended together regardless of their order;
   *  labels are drawn on top.
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
   *  The tooltip is an object with the following fields: `x`, `y`, `category`, `text`, `identifier`.
   *  To listen for a tooltip change, use `onTooltip`. */
  tooltip?: DataPoint | null;

  /** The current single or multiple point selection.
   *  Selection is triggered by clicking on the points (shift/cmd+click will toggle points).
   *  The selection is an array of objects with the following fields: `x`, `y`, `category`, `text`, `identifier`.
   *  To listen to selection change, use `onSelection`. */
  selection?: DataPoint[] | null;

  /** A rectangle or a polygon (list of points) that represents the range selection.
   *  If the value is a list of points, it is interpreted as a lasso selection
   *  with a closed polygon with the list of points as vertices. */
  rangeSelection?: Rectangle | null;

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

  /** An async function that returns a data point near the given (x, y) location.
   *  The `unitDistance` parameter is the distance of a single pixel in data domain.
   *  You can use this to determine the distance threshold for selecting a point. */
  querySelection?: ((x: number, y: number, unitDistance: number) => Promise<DataPoint | null>) | null;

  /** An async function that returns labels for a list of clusters.
   *  Each cluster is given as a list of rectangles that approximately cover the region. */
  queryClusterLabels?: ((clusters: Rectangle[][]) => Promise<(LabelContent | null)[]>) | null;

  /** A custom renderer to draw the tooltip content. */
  customTooltip?: CustomComponent<HTMLDivElement, { tooltip: DataPoint }> | null;

  /** A custom renderer to draw overlay on top of the embedding view. */
  customOverlay?: CustomComponent<HTMLDivElement, { proxy: OverlayProxy }> | null;

  /** A cache for intermediate results, such as automatic labels. The data isn't part of the cache keys
   *  (the view can't identify its data across sessions), so use a cache that's specific to the data. */
  cache?: Cache | null;
}

export class EmbeddingView {
  private component: any;
  private currentProps: EmbeddingViewProps;

  constructor(target: HTMLElement, props: EmbeddingViewProps) {
    this.currentProps = { ...props };
    this.component = createClassComponent({ component: Component, target: target, props: props });
  }

  update(props: Partial<EmbeddingViewProps>) {
    let updates: Partial<EmbeddingViewProps> = {};
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
