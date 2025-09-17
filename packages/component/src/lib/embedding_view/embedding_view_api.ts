// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createClassComponent } from "svelte/legacy";

import Component from "./EmbeddingView.svelte";

import type { Point, Rectangle, ViewportState } from "../utils.js";
import type { EmbeddingViewConfig } from "./embedding_view_config.js";
import type { ThemeConfig } from "./theme.js";
import type { Cache, CustomComponent, DataPoint, Label, OverlayProxy } from "./types.js";

export interface EmbeddingViewProps {
  /** The data. */
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
   *  Each label must have `x`, `y`, and `text` properties,
   *  and optionally `level` and `priority`. */
  labels?: Label[] | null;

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

  /** An async function that returns a data point near the given (x, y) location.
   *  The `unitDistance` parameter is the distance of a single pixel in data domain.
   *  You can use this to determine the distance threshold for selecting a point. */
  querySelection?: ((x: number, y: number, unitDistance: number) => Promise<DataPoint | null>) | null;

  /** An async function that returns labels for a list of clusters.
   *  Each cluster is given as a list of rectangles that approximately cover the region. */
  queryClusterLabels?: ((clusters: Rectangle[][]) => Promise<(string | null)[]>) | null;

  /** A custom renderer to draw the tooltip content. */
  customTooltip?: CustomComponent<HTMLDivElement, { tooltip: DataPoint }> | null;

  /** A custom renderer to draw overlay on top of the embedding view. */
  customOverlay?: CustomComponent<HTMLDivElement, { proxy: OverlayProxy }> | null;

  /** A cache for intermediate results. */
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
