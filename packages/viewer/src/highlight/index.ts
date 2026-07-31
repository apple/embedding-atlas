// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export { highlight } from "./action.js";
export type { HighlightOptions, HighlightScheme, HighlightScorer, HighlightSpan } from "./action.js";
export { embeddingHighlightScorer, type EmbeddingScorerOptions } from "./embedding-scorer.js";
export { fuzzyHighlightScorer, fuzzyMatch, type FuzzyMatchOptions } from "./fuzzy-scorer.js";
export { defaultScheme, highlightApiSupported, type HighlightInterpolator } from "./registry.js";
export { extractSegments, rangeForSegment, type SegmentNode, type TextSegment } from "./text-map.js";
