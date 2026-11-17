// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export {
  EmbeddingView,
  EmbeddingViewMosaic,
  maxDensityModeCategories,
  type EmbeddingViewMosaicProps,
  type EmbeddingViewProps,
} from "./embedding_view/api.js";

export { defaultCategoryColors } from "./colors.js";
export {
  categoryIndexExpr,
  DEFAULT_NULL_COLOR,
  DEFAULT_OTHER_COLOR,
  fieldExpr,
  fieldKey,
  MAX_SCALE_CATEGORIES,
  scaleColors,
  type ColorScale,
  type SQLField,
} from "./embedding_view/data/fields.js";

export { ParallelCoordinatesView, type ParallelCoordinatesViewProps } from "./parallel_coordinates_view/api.js";
export type {
  ParallelCoordinatesViewTheme,
  ParallelCoordinatesViewThemeConfig,
} from "./parallel_coordinates_view/theme.js";

export type { EmbeddingViewConfig } from "./embedding_view/embedding_view_config.js";
export type { EmbeddingViewTheme } from "./embedding_view/theme.js";
export type {
  CustomComponent,
  DataField,
  DataPoint,
  DataPointID,
  Label,
  LabelContent,
  OverlayProxy,
} from "./embedding_view/types.js";
export type { Point, Rectangle, ViewportState } from "./utils.js";
