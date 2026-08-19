// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export { EmbeddingView, type EmbeddingViewProps } from "./embedding_view_api.js";
export { EmbeddingViewMosaic, type EmbeddingViewMosaicProps } from "./embedding_view_mosaic_api.js";

export function maxDensityModeCategories(): number {
  // Technically we can support 256 categories, but it's limited by speed and memory usage.
  // 32 is chosen here so that the total memory usage of a 2048 x 2048 x categories x f16 buffer is 256MB.
  return 32;
}
