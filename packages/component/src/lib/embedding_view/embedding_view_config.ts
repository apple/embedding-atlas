// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { ThemeConfig } from "./theme.js";

export interface EmbeddingViewConfig {
  /** Color scheme. */
  colorScheme?: "light" | "dark" | null;

  /** Theme configuration for customizing colors, fonts, and other visual properties. */
  theme?: ThemeConfig | null;

  /** View mode. */
  mode?: "points" | "density" | null;

  /** Minimum average density for density contours to show up.
   * The density is measured as number of points per square points (aka., px in CSS units). */
  minimumDensity?: number | null;

  /** Override the automatically calculated point size.
   * If not specified, point size is calculated based on density. */
  pointSize?: number | null;

  /** Generate labels automatically.
   * By default labels are generated automatically if the `labels` prop is not specified,
   * and a `text` column is specified in the Mosaic view,
   * or a `queryClusterLabels` callback is specified in the non-Mosaic view.
   * Set this to `false` to disable automatic labels. */
  autoLabelEnabled?: boolean | null;

  /** The density threshold to filter the clusters before generating automatic labels.
   * The value is relative to the max density. */
  autoLabelDensityThreshold?: number | null;

  /** The stop words for automatic label generation. By default use NLTK stop words. */
  autoLabelStopWords?: string[] | null;
}
