// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { EmbeddingViewConfig } from "@embedding-atlas/component";

export interface EmbeddingSpec {
  type: "embedding";
  title: string;
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
