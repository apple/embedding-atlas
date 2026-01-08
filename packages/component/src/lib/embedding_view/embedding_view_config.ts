// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export interface EmbeddingViewConfig {
  /** Color scheme. */
  colorScheme?: "light" | "dark" | null;

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

  /** Approximate maximum number of points to render when downsampling is active.
   * Points are sampled with bias toward sparse regions (fewer points kept in dense areas).
   * The sampling probability is given by this formula:
   * P(i) = (downsampleMaxPoints / numPointsInViewport) * (2 / (1 + density(p_i) / maxDensity * downsampleDensityWeight))
   * Default: 4,000,000. Set to null or Infinity to disable downsampling. */
  downsampleMaxPoints?: number | null;

  /** Density weight for downsampling (0-10).
   * Higher values mean more aggressive culling in dense areas.
   * Default: 5 */
  downsampleDensityWeight?: number | null;
}
