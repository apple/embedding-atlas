// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Camera3DState, EmbeddingViewConfig, Point, Rectangle, ViewportState } from "@embedding-atlas/component";

export interface EmbeddingSpec {
  type: "embedding";
  title?: string;

  data: {
    x: string;
    y: string;
    z?: string | null;
    text?: string | null;
    image?: string | null;
    importance?: string | null;
    category?: string | null;
    neighbors?: string | null;
  };

  mode?: "points" | "density" | "points-3d";
  minimumDensity?: number;
  pointSize?: number;
  /** Maximum number of points to render (for downsampling). Default: 4000000. Set to null to disable. */
  downsampleMaxPoints?: number | null;
  config?: EmbeddingViewConfig;
}

export interface EmbeddingState {
  /** The viewport state */
  viewport?: ViewportState;
  /** The 3D camera state (used in points-3d mode). */
  camera3D?: Camera3DState;
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
