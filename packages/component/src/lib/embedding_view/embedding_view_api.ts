// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createClassComponent } from "svelte/legacy";

import Component from "./EmbeddingView.svelte";

import type { Point, Rectangle, ViewportState } from "../utils.js";
import type { Theme } from "./theme.js";
import type { AutomaticLabelsConfig, CustomComponent, DataPoint, OverlayProxy } from "./types.js";

export interface EmbeddingViewProps {
  /** The data. */
  data: {
    x: Float32Array;
    y: Float32Array;
    category?: Uint8Array | null;
  };

  /** The colors for the categories. */
  categoryColors?: string[] | null;

  /** The tooltip. Tooltip is triggered on hover. */
  tooltip?: DataPoint | null;

  /** The selection. Selection is triggered with click or shift/cmd-click. */
  selection?: DataPoint[] | null;

  /** The range selection. Shift-drag to create a range selection. */
  rangeSelection?: Rectangle | null;

  /** The width of the view. */
  width?: number | null;

  /** The height of the view. */
  height?: number | null;

  /** The pixel ratio of the view. */
  pixelRatio?: number | null;

  /** Color scheme. */
  colorScheme: "light" | "dark" | null;

  /** Theme. */
  theme?: Theme | null;

  /** The viewport state. */
  viewportState?: ViewportState | null;

  /** Set to true to automatically create labels from the text field. */
  automaticLabels?: AutomaticLabelsConfig | boolean | null;

  /** View mode. */
  mode?: "points" | "density" | null;

  /** Minimum average density for density contours to show up.
   * The density is measured as number of points per square points (aka., px in CSS units).
   */
  minimumDensity?: number | null;

  /** A custom renderer to draw the tooltip content. */
  customTooltip?: CustomComponent<HTMLDivElement, { tooltip: DataPoint }> | null;

  /** A custom renderer to draw overlay on top of the embedding view. */
  customOverlay?: CustomComponent<HTMLDivElement, { proxy: OverlayProxy }> | null;

  /** A function to query selected point given (x, y) location, and a unit distance (distance of 1pt in data units). */
  querySelection?: ((x: number, y: number, unitDistance: number) => Promise<DataPoint | null>) | null;

  /** A function that returns summary labels for clusters. Each cluster is given by a list of rectangles that approximate its shape. */
  queryClusterLabels?: ((clusters: Rectangle[][]) => Promise<(string | null)[]>) | null;

  /** A callback for when viewportState changes. */
  onViewportState?: ((value: ViewportState) => void) | null;

  /** A callback for when tooltip changes. */
  onTooltip?: ((value: DataPoint | null) => void) | null;

  /** A callback for when selection changes. */
  onSelection?: ((value: DataPoint[] | null) => void) | null;

  /** A callback for when rangeSelection changes. */
  onRangeSelection?: ((value: Rectangle | Point[] | null) => void) | null;
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
