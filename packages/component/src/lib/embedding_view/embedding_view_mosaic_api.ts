// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator, Selection } from "@uwdata/mosaic-core";
import { createClassComponent } from "svelte/legacy";

import Component from "./EmbeddingViewMosaic.svelte";

import type { Point, Rectangle, ViewportState } from "../utils.js";
import type { EmbeddingViewConfig } from "./embedding_view_config.js";
import type { ThemeConfig } from "./theme.js";
import type { Cache, CustomComponent, DataField, DataPoint, DataPointID, Label, OverlayProxy } from "./types.js";

export interface EmbeddingViewMosaicProps {
  /** The Mosaic coordinator.
   *  If not specified, the default coordinator from Mosaic's `coordinator()` method will be used. */
  coordinator?: Coordinator;

  /** The data table name. */
  table: string;

  /** The x column name. */
  x: string;

  /** The y column name. */
  y: string;

  /** The name of the category column.
   *  The categories should be represented as integers starting from 0.
   *  If you have categories represented as strings, you should first convert them to 0-indexed integers. */
  category?: string | null;

  /** The name of the text column.
   *  If specified, the default tooltip shows the text content.
   *  The text content is also used to generate labels automatically. */
  text?: string | null;

  /** The name of the identifier (aka., id) column.
   *  If specified, the `selection` object will contain an `identifier` property that you can use to identify the point. */
  identifier?: string | null;

  /** Additional fields for the tooltip data element.
   *  Each field can be specified as a column name or a SQL expression. */
  additionalFields?: Record<string, DataField> | null;

  /** The colors for the categories.
   *  Category `i` will use the `i`-th color from this list.
   *  If not specified, default colors will be used. */
  categoryColors?: string[] | null;

  /** A Mosaic `Selection` object to filter the contents of this view. */
  filter?: Selection | null;

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

  /** A custom renderer to draw the tooltip content. */
  customTooltip?: CustomComponent<HTMLDivElement, { tooltip: DataPoint }> | null;

  /** A custom renderer to draw overlay on top of the embedding view. */
  customOverlay?: CustomComponent<HTMLDivElement, { proxy: OverlayProxy }> | null;

  /** A cache for intermediate results. */
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
