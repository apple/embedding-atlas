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
    /** The Z column. When set, the embedding view renders as a navigable 3D point cloud. */
    z?: string | null;
    text?: string | null;
    image?: string | null;
    importance?: string | null;
    category?: string | null;
    neighbors?: string | null;
  };

  mode?: "points" | "density";
  minimumDensity?: number;
  pointSize?: number;
  /** Maximum number of points to render (for downsampling). Default: 4000000. Set to null to disable. */
  downsampleMaxPoints?: number | null;
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
  /** Overrides the default display mode. When `data.z` is set, the view renders in 3D unless
   *  this is explicitly set to `"2d"`; when `data.z` is unset, this has no effect (there's no
   *  3D data to render). */
  displayMode?: "2d" | "3d";
}
