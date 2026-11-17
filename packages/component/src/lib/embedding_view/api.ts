// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export { EmbeddingView, type EmbeddingViewProps } from "./embedding_view_api.js";
export { EmbeddingViewMosaic, type EmbeddingViewMosaicProps } from "./embedding_view_mosaic_api.js";

import { MAX_DENSITY_CATEGORIES } from "./layers.js";

export function maxDensityModeCategories(): number {
  return MAX_DENSITY_CATEGORIES;
}
