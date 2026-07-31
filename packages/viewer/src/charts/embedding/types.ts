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
    neighbors?: string | null;
  };

  /** @deprecated Superseded by `layers`. "density" turns on the density layer; kept for previously saved states. */
  mode?: "points" | "density";

  /** Visibility of the view's layers. Missing entries default to points and labels on, density off,
   * or to the equivalent of the deprecated `mode` field when it is present. */
  layers?: { points?: boolean; density?: boolean; labels?: boolean };

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
