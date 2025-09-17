// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export {
  EmbeddingView,
  EmbeddingViewMosaic,
  maxDensityModeCategories,
  type EmbeddingViewMosaicProps,
  type EmbeddingViewProps,
} from "./embedding_view/api.js";

export { defaultCategoryColors } from "./colors.js";

export type { EmbeddingViewConfig } from "./embedding_view/embedding_view_config.js";
export type { EmbeddingViewTheme } from "./embedding_view/theme.js";
export type {
  CustomComponent,
  DataField,
  DataPoint,
  DataPointID,
  Label,
  OverlayProxy,
} from "./embedding_view/types.js";
export type { Point, Rectangle, ViewportState } from "./utils.js";
