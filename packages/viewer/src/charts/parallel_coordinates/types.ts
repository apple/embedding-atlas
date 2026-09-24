// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Scale, SQLField, SQLTable } from "../spec/spec.js";

export interface ParallelCoordinatesSpec {
  type: "parallel-coordinates";

  /** The title of the chart */
  title?: string;

  data: {
    /** Data source, default to the main data table */
    from?: SQLTable;

    /** Filter the data. Use $filter to refer to the shared filter (a cross-filter) */
    filter?: "$filter";

    /** The data fields */
    fields: SQLField[];

    /** Color lines by the given field */
    color?: SQLField;
  };

  /** Opacity of the lines, default 0.8 */
  opacity?: number;

  /** Fixed chart width in pixels. When unset, the chart fills the width given by the layout. */
  width?: number;

  /** Fixed chart height in pixels. When unset, the chart fills the height given by the layout. */
  height?: number;

  /** Specify color scale */
  scales?: {
    color?: Scale;
  };
}

export interface ParallelCoordinatesState {
  /**
   * Per-axis brush selection, indexed like `data.fields`. Each entry is `[from, to]` in normalized
   * `[0, 1]` coordinates (0 = top), or `null` for no brush on that axis.
   */
  brush?: ([number, number] | null)[];
}
