// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Media handling module for qualitative coding tool.
 * Provides utilities for image/video detection, frame extraction, and embedding.
 */

// Types and utilities
export {
  type MediaType,
  type VideoSourceType,
  type MediaMetadata,
  type VideoFrame,
  type FrameExtractionOptions,
  type VideoProcessingOptions,
  type EmbeddingAggregation,
  defaultFrameExtractionOptions,
  defaultVideoProcessingOptions,
  detectMediaType,
  isYouTubeUrl,
  isVimeoUrl,
  extractYouTubeId,
  extractVimeoId,
  getVideoSourceType,
  getYouTubeThumbnail,
  getYouTubeEmbedUrl,
  fetchImageAsDataUrl,
  getImageDimensions,
  createThumbnail,
  aggregateEmbeddings,
  isDataUrl,
  isHttpUrl,
  isImageValue,
  isVideoValue,
} from "./types.js";

// Frame extraction
export {
  type FrameExtractionProgressCallback,
  type FrameExtractionResult,
  extractFramesFromUrl,
  extractFramesFromFile,
  getVideoMetadata,
  extractSingleFrame,
  extractKeyframes,
  createVideoThumbnail,
} from "./frame_extractor.js";

// Components
export { default as VideoProcessingSettings } from "./VideoProcessingSettings.svelte";
export { default as MediaPreview } from "./MediaPreview.svelte";
