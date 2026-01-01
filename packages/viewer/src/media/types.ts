// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Media handling types and utilities for qualitative coding tool.
 * Supports images, videos (local and streaming), and frame extraction.
 */

/** Supported media types */
export type MediaType = "image" | "video" | "audio" | "unknown";

/** Video source type */
export type VideoSourceType = "youtube" | "vimeo" | "direct" | "local";

/** Media metadata */
export interface MediaMetadata {
  type: MediaType;
  mimeType?: string;
  width?: number;
  height?: number;
  duration?: number; // seconds
  frameRate?: number;
  fileSize?: number;
  fileName?: string;
  sourceUrl?: string;
}

/** Video frame */
export interface VideoFrame {
  timestamp: number; // seconds
  dataUrl: string; // base64 image data URL
  width: number;
  height: number;
}

/** Frame extraction options */
export interface FrameExtractionOptions {
  /** Frames per second to extract (default: 1) */
  fps: number;
  /** Maximum number of frames to extract (default: 100) */
  maxFrames: number;
  /** Start time in seconds (default: 0) */
  startTime: number;
  /** End time in seconds (default: video duration) */
  endTime?: number;
  /** Output image format (default: "jpeg") */
  format: "jpeg" | "png";
  /** JPEG quality 0-100 (default: 80) */
  quality: number;
  /** Scale factor for output frames (default: 1.0) */
  scale: number;
}

/** Default frame extraction options */
export const defaultFrameExtractionOptions: FrameExtractionOptions = {
  fps: 1,
  maxFrames: 100,
  startTime: 0,
  format: "jpeg",
  quality: 80,
  scale: 1.0,
};

/** Video embedding aggregation method */
export type EmbeddingAggregation = "mean" | "max" | "attention_weighted" | "temporal_pooling";

/** Video processing options */
export interface VideoProcessingOptions {
  frameExtraction: FrameExtractionOptions;
  aggregation: EmbeddingAggregation;
  /** Store individual frame embeddings (allows frame-level coding) */
  storeFrameEmbeddings: boolean;
}

/** Default video processing options */
export const defaultVideoProcessingOptions: VideoProcessingOptions = {
  frameExtraction: defaultFrameExtractionOptions,
  aggregation: "mean",
  storeFrameEmbeddings: false,
};

// ============================================================================
// URL Detection Utilities
// ============================================================================

/** Image file extensions */
const IMAGE_EXTENSIONS = /\.(jpg|jpeg|png|gif|webp|svg|bmp|ico|tiff?)$/i;

/** Video file extensions */
const VIDEO_EXTENSIONS = /\.(mp4|webm|ogg|mov|avi|mkv|m4v|flv|wmv)$/i;

/** Audio file extensions */
const AUDIO_EXTENSIONS = /\.(mp3|wav|ogg|flac|aac|m4a|wma)$/i;

/** YouTube URL patterns */
const YOUTUBE_PATTERNS = [
  /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
  /youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})/,
];

/** Vimeo URL patterns */
const VIMEO_PATTERNS = [
  /vimeo\.com\/(\d+)/,
  /player\.vimeo\.com\/video\/(\d+)/,
];

/**
 * Detect media type from URL or file extension
 */
export function detectMediaType(value: any): MediaType {
  if (value == null) return "unknown";

  // Handle data URLs
  if (typeof value === "string") {
    if (value.startsWith("data:image/")) return "image";
    if (value.startsWith("data:video/")) return "video";
    if (value.startsWith("data:audio/")) return "audio";

    // Check file extensions
    if (IMAGE_EXTENSIONS.test(value)) return "image";
    if (VIDEO_EXTENSIONS.test(value)) return "video";
    if (AUDIO_EXTENSIONS.test(value)) return "audio";

    // Check video streaming services
    if (isYouTubeUrl(value) || isVimeoUrl(value)) return "video";
  }

  // Handle binary data (check magic bytes)
  if (value?.bytes instanceof Uint8Array) {
    const bytes = value.bytes;
    if (isImageMagicBytes(bytes)) return "image";
    if (isVideoMagicBytes(bytes)) return "video";
  }

  return "unknown";
}

/**
 * Check if URL is a YouTube video
 */
export function isYouTubeUrl(url: string): boolean {
  return YOUTUBE_PATTERNS.some((pattern) => pattern.test(url));
}

/**
 * Check if URL is a Vimeo video
 */
export function isVimeoUrl(url: string): boolean {
  return VIMEO_PATTERNS.some((pattern) => pattern.test(url));
}

/**
 * Extract YouTube video ID from URL
 */
export function extractYouTubeId(url: string): string | null {
  for (const pattern of YOUTUBE_PATTERNS) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

/**
 * Extract Vimeo video ID from URL
 */
export function extractVimeoId(url: string): string | null {
  for (const pattern of VIMEO_PATTERNS) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

/**
 * Get video source type from URL
 */
export function getVideoSourceType(url: string): VideoSourceType {
  if (isYouTubeUrl(url)) return "youtube";
  if (isVimeoUrl(url)) return "vimeo";
  if (url.startsWith("blob:") || url.startsWith("file:")) return "local";
  return "direct";
}

/**
 * Get YouTube thumbnail URL
 */
export function getYouTubeThumbnail(videoId: string, quality: "default" | "hq" | "mq" | "sd" | "maxres" = "hq"): string {
  const qualityMap = {
    default: "default",
    mq: "mqdefault",
    hq: "hqdefault",
    sd: "sddefault",
    maxres: "maxresdefault",
  };
  return `https://img.youtube.com/vi/${videoId}/${qualityMap[quality]}.jpg`;
}

/**
 * Get YouTube embed URL
 */
export function getYouTubeEmbedUrl(videoId: string): string {
  return `https://www.youtube.com/embed/${videoId}`;
}

// ============================================================================
// Magic Bytes Detection
// ============================================================================

/**
 * Check if bytes represent an image file
 */
function isImageMagicBytes(bytes: Uint8Array): boolean {
  if (bytes.length < 4) return false;

  // PNG
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return true;
  }

  // JPEG
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return true;
  }

  // GIF
  if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46) {
    return true;
  }

  // WebP
  if (bytes.length >= 12 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
    if (bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) {
      return true;
    }
  }

  // BMP
  if (bytes[0] === 0x42 && bytes[1] === 0x4d) {
    return true;
  }

  return false;
}

/**
 * Check if bytes represent a video file
 */
function isVideoMagicBytes(bytes: Uint8Array): boolean {
  if (bytes.length < 12) return false;

  // MP4/MOV (ftyp box)
  if (bytes[4] === 0x66 && bytes[5] === 0x74 && bytes[6] === 0x79 && bytes[7] === 0x70) {
    return true;
  }

  // WebM/MKV (EBML header)
  if (bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3) {
    return true;
  }

  // AVI (RIFF...AVI)
  if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
    if (bytes[8] === 0x41 && bytes[9] === 0x56 && bytes[10] === 0x49) {
      return true;
    }
  }

  return false;
}

// ============================================================================
// Media Loading Utilities
// ============================================================================

/**
 * Fetch an image from URL and return as data URL
 */
export async function fetchImageAsDataUrl(url: string): Promise<string> {
  const response = await fetch(url, { mode: "cors" });
  if (!response.ok) {
    throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
  }

  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Load image and get dimensions
 */
export async function getImageDimensions(src: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = reject;
    img.src = src;
  });
}

/**
 * Create a thumbnail from an image
 */
export async function createThumbnail(
  src: string,
  maxWidth: number = 256,
  maxHeight: number = 256
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement("canvas");
      let { width, height } = img;

      // Calculate scaled dimensions
      if (width > maxWidth || height > maxHeight) {
        const ratio = Math.min(maxWidth / width, maxHeight / height);
        width = Math.round(width * ratio);
        height = Math.round(height * ratio);
      }

      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Failed to get canvas context"));
        return;
      }

      ctx.drawImage(img, 0, 0, width, height);
      resolve(canvas.toDataURL("image/jpeg", 0.8));
    };
    img.onerror = reject;
    img.src = src;
  });
}

// ============================================================================
// Embedding Aggregation Utilities
// ============================================================================

/**
 * Aggregate frame embeddings into a single embedding
 */
export function aggregateEmbeddings(
  embeddings: Float32Array[],
  method: EmbeddingAggregation
): Float32Array {
  if (embeddings.length === 0) {
    throw new Error("No embeddings to aggregate");
  }

  const dim = embeddings[0].length;

  switch (method) {
    case "mean":
      return meanPooling(embeddings);
    case "max":
      return maxPooling(embeddings);
    case "temporal_pooling":
      return temporalPooling(embeddings);
    case "attention_weighted":
      // Fall back to mean for now - attention would need learned weights
      return meanPooling(embeddings);
    default:
      return meanPooling(embeddings);
  }
}

/**
 * Mean pooling: average all embeddings
 */
function meanPooling(embeddings: Float32Array[]): Float32Array {
  const dim = embeddings[0].length;
  const result = new Float32Array(dim);

  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      result[i] += emb[i];
    }
  }

  const n = embeddings.length;
  for (let i = 0; i < dim; i++) {
    result[i] /= n;
  }

  return result;
}

/**
 * Max pooling: take maximum value for each dimension
 */
function maxPooling(embeddings: Float32Array[]): Float32Array {
  const dim = embeddings[0].length;
  const result = new Float32Array(dim).fill(-Infinity);

  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      result[i] = Math.max(result[i], emb[i]);
    }
  }

  return result;
}

/**
 * Temporal pooling: weighted average with more weight on middle frames
 */
function temporalPooling(embeddings: Float32Array[]): Float32Array {
  const dim = embeddings[0].length;
  const n = embeddings.length;
  const result = new Float32Array(dim);

  // Gaussian-like weights centered in the middle
  const weights: number[] = [];
  const center = (n - 1) / 2;
  const sigma = n / 4;
  let weightSum = 0;

  for (let i = 0; i < n; i++) {
    const w = Math.exp(-Math.pow(i - center, 2) / (2 * sigma * sigma));
    weights.push(w);
    weightSum += w;
  }

  for (let i = 0; i < n; i++) {
    const w = weights[i] / weightSum;
    for (let j = 0; j < dim; j++) {
      result[j] += embeddings[i][j] * w;
    }
  }

  return result;
}

// ============================================================================
// Data URL Utilities
// ============================================================================

/**
 * Check if a string is a valid data URL
 */
export function isDataUrl(value: string): boolean {
  return typeof value === "string" && value.startsWith("data:");
}

/**
 * Check if a string is a valid HTTP(S) URL
 */
export function isHttpUrl(value: string): boolean {
  return typeof value === "string" && (value.startsWith("http://") || value.startsWith("https://"));
}

/**
 * Check if a value represents an image
 */
export function isImageValue(value: any): boolean {
  return detectMediaType(value) === "image";
}

/**
 * Check if a value represents a video
 */
export function isVideoValue(value: any): boolean {
  return detectMediaType(value) === "video";
}
