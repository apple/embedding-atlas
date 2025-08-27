// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export {
  EmbeddingView,
  EmbeddingViewMosaic,
  maxDensityModeCategories,
  type EmbeddingViewMosaicProps,
  type EmbeddingViewProps,
} from "./embedding_view/api.js";

export { defaultCategoryColors } from "./colors.js";

export type { Theme } from "./embedding_view/theme.js";
export type {
  AutomaticLabelsConfig,
  CustomComponent,
  DataField,
  DataPoint,
  DataPointID,
  OverlayProxy,
} from "./embedding_view/types.js";
export type { Point, Rectangle, ViewportState } from "./utils.js";
