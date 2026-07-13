// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export {
  EmbeddingView,
  EmbeddingView3D,
  EmbeddingView3DMosaic,
  EmbeddingViewMosaic,
  maxDensityModeCategories,
  type EmbeddingView3DMosaicProps,
  type EmbeddingView3DProps,
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
  LabelContent,
  OverlayProxy,
} from "./embedding_view/types.js";
export type { Point, Rectangle, ViewportState } from "./utils.js";
