// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { EmbeddingViewConfig, Point, Rectangle, ViewportState } from "@embedding-atlas/component";

export interface EmbeddingSpec {
  type: "embedding";
  title?: string;

  data: {
    /** The data table, leave undefined for the default table. */
    table?: string;

    /** The row id column, must be set when using a non-default table. */
    id?: string;

    x: string;
    y: string;
    text?: string | null;
    image?: string | null;
    importance?: string | null;
    category?: string | null;
    isGis?: boolean;
    /** Axis-aligned bounds for (x, y). When set, the scatter query can pack
     *  coordinates on the wire — see `EmbeddingViewMosaic`. */
    bounds?: { x: [number, number]; y: [number, number] } | null;
    /** Names of pre-computed u16-quantised x/y columns on the source
     *  table. When set, the scatter query is a pure scan — no per-row
     *  arithmetic. The loader fills these at CTAS time. */
    precomputed?: { x_u16: string; y_u16: string; y_is_mercator?: boolean } | null;
    /** Optional viewport defaults derived from the dataset bounds. When
     *  present, the embedding view uses these for its initial centerX /
     *  centerY / scaler instead of running APPROX_QUANTILE+STDDEV — saves
     *  ~5 s of cold-load time on 75 M-row GIS datasets. Unlike ``bounds``
     *  this is never used to pack the wire payload, so quantisation stays
     *  off. */
    viewportHint?: {
      centerX: number;
      centerY: number;
      rangeX: number;
      rangeY: number;
      rowCount?: number;
      skipDeferredRefine?: boolean;
    } | null;
    neighbors?: string | null;
  };

  mode?: "points" | "density";
  mapStyle?: string | null;
  minimumDensity?: number;
  pointSize?: number;
  /** Maximum number of points to render (for downsampling). Default: 4000000. Set to null to disable. */
  downsampleMaxPoints?: number | null;
  /** Max points rendered while the user is actively zooming/panning. Default: 200000. */
  downsampleMaxPointsInteractive?: number | null;
  config?: EmbeddingViewConfig;
}

export interface EmbeddingState {
  /** The viewport state */
  viewport?: ViewportState;
  /** State of the legend */
  legend?: {
    /** Selected categories */
    selection?: string[];
  };
  /**
   * State of the brush selection. Can be a rectangle or a list of points for a lasso selection.
   * Coordinates should be in data units.
   */
  brush?: Rectangle | Point[];
}
