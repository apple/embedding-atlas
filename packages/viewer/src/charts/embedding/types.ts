// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { EmbeddingViewConfig, Point, Rectangle, ViewportState } from "@embedding-atlas/component";

export interface EmbeddingSpec {
  type: "embedding";
  title?: string;

  data: {
    x: string;
    y: string;
    text?: string | null;
    category?: string | null;
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
}
