// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/**
 * Video frame extraction using HTML5 Video and Canvas APIs.
 *
 * This module provides browser-based frame extraction without requiring
 * external dependencies like ffmpeg.wasm. It uses the native video element
 * to seek to specific timestamps and captures frames using canvas.
 */

import {
  type VideoFrame,
  type FrameExtractionOptions,
  type MediaMetadata,
  defaultFrameExtractionOptions,
  isYouTubeUrl,
  isVimeoUrl,
} from "./types.js";

/**
 * Progress callback for frame extraction
 */
export type FrameExtractionProgressCallback = (
  current: number,
  total: number,
  message: string
) => void;

/**
 * Frame extraction result
 */
export interface FrameExtractionResult {
  frames: VideoFrame[];
  metadata: MediaMetadata;
  success: boolean;
  error?: string;
}

/**
 * Extract frames from a video URL using HTML5 Video element
 */
export async function extractFramesFromUrl(
  url: string,
  options: Partial<FrameExtractionOptions> = {},
  onProgress?: FrameExtractionProgressCallback
): Promise<FrameExtractionResult> {
  const opts = { ...defaultFrameExtractionOptions, ...options };

  // Check for streaming services - they need different handling
  if (isYouTubeUrl(url) || isVimeoUrl(url)) {
    return {
      frames: [],
      metadata: { type: "video", sourceUrl: url },
      success: false,
      error: "YouTube and Vimeo videos cannot be frame-extracted directly. Use yt-dlp backend for these.",
    };
  }

  try {
    onProgress?.(0, 1, "Loading video...");

    // Create video element
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.playsInline = true;

    // Load video
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error(`Failed to load video: ${url}`));
      video.src = url;
    });

    const duration = video.duration;
    const width = Math.round(video.videoWidth * opts.scale);
    const height = Math.round(video.videoHeight * opts.scale);

    const metadata: MediaMetadata = {
      type: "video",
      width: video.videoWidth,
      height: video.videoHeight,
      duration,
      sourceUrl: url,
    };

    // Calculate frame timestamps
    const startTime = opts.startTime;
    const endTime = opts.endTime ?? duration;
    const interval = 1 / opts.fps;
    const timestamps: number[] = [];

    for (let t = startTime; t < endTime && timestamps.length < opts.maxFrames; t += interval) {
      timestamps.push(t);
    }

    onProgress?.(0, timestamps.length, `Extracting ${timestamps.length} frames...`);

    // Create canvas for frame capture
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    if (!ctx) {
      throw new Error("Failed to create canvas context");
    }

    const frames: VideoFrame[] = [];

    // Extract frames
    for (let i = 0; i < timestamps.length; i++) {
      const timestamp = timestamps[i];

      // Seek to timestamp
      await seekToTime(video, timestamp);

      // Draw frame to canvas
      ctx.drawImage(video, 0, 0, width, height);

      // Convert to data URL
      const mimeType = opts.format === "png" ? "image/png" : "image/jpeg";
      const quality = opts.format === "jpeg" ? opts.quality / 100 : undefined;
      const dataUrl = canvas.toDataURL(mimeType, quality);

      frames.push({
        timestamp,
        dataUrl,
        width,
        height,
      });

      onProgress?.(i + 1, timestamps.length, `Extracted frame ${i + 1}/${timestamps.length}`);
    }

    // Clean up
    video.src = "";
    video.load();

    return {
      frames,
      metadata,
      success: true,
    };
  } catch (error) {
    return {
      frames: [],
      metadata: { type: "video", sourceUrl: url },
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Extract frames from a File or Blob
 */
export async function extractFramesFromFile(
  file: File | Blob,
  options: Partial<FrameExtractionOptions> = {},
  onProgress?: FrameExtractionProgressCallback
): Promise<FrameExtractionResult> {
  const url = URL.createObjectURL(file);
  try {
    const result = await extractFramesFromUrl(url, options, onProgress);
    if (file instanceof File) {
      result.metadata.fileName = file.name;
      result.metadata.fileSize = file.size;
    }
    return result;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Seek video to a specific timestamp and wait for it to be ready
 */
async function seekToTime(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error(`Seek timeout at ${time}s`));
    }, 10000);

    const onSeeked = () => {
      clearTimeout(timeout);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
      resolve();
    };

    const onError = () => {
      clearTimeout(timeout);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
      reject(new Error(`Seek error at ${time}s`));
    };

    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    video.currentTime = time;
  });
}

/**
 * Get video metadata without extracting frames
 */
export async function getVideoMetadata(url: string): Promise<MediaMetadata> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.preload = "metadata";

    video.onloadedmetadata = () => {
      resolve({
        type: "video",
        width: video.videoWidth,
        height: video.videoHeight,
        duration: video.duration,
        sourceUrl: url,
      });
      video.src = "";
    };

    video.onerror = () => {
      reject(new Error(`Failed to load video metadata: ${url}`));
    };

    video.src = url;
  });
}

/**
 * Extract a single frame at a specific timestamp
 */
export async function extractSingleFrame(
  url: string,
  timestamp: number,
  options: Partial<Pick<FrameExtractionOptions, "format" | "quality" | "scale">> = {}
): Promise<VideoFrame | null> {
  const result = await extractFramesFromUrl(url, {
    ...options,
    fps: 1000, // High FPS to get exactly the frame we want
    maxFrames: 1,
    startTime: timestamp,
    endTime: timestamp + 0.001,
  });

  return result.frames[0] ?? null;
}

/**
 * Extract keyframes from a video (scene changes)
 *
 * This is a simplified implementation that samples at regular intervals.
 * A more sophisticated implementation would analyze frame differences.
 */
export async function extractKeyframes(
  url: string,
  options: {
    maxFrames?: number;
    format?: "jpeg" | "png";
    quality?: number;
    scale?: number;
  } = {},
  onProgress?: FrameExtractionProgressCallback
): Promise<FrameExtractionResult> {
  const { maxFrames = 10, format = "jpeg", quality = 80, scale = 1.0 } = options;

  try {
    // Get video duration first
    const metadata = await getVideoMetadata(url);
    if (!metadata.duration) {
      throw new Error("Could not determine video duration");
    }

    // Sample evenly distributed frames
    const interval = metadata.duration / (maxFrames + 1);
    const timestamps: number[] = [];
    for (let i = 1; i <= maxFrames; i++) {
      timestamps.push(interval * i);
    }

    // Calculate FPS that would give us exactly these frames
    // We'll use a custom extraction approach instead
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.playsInline = true;

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error(`Failed to load video: ${url}`));
      video.src = url;
    });

    const width = Math.round(video.videoWidth * scale);
    const height = Math.round(video.videoHeight * scale);

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    if (!ctx) {
      throw new Error("Failed to create canvas context");
    }

    const frames: VideoFrame[] = [];

    for (let i = 0; i < timestamps.length; i++) {
      const timestamp = timestamps[i];

      await seekToTime(video, timestamp);
      ctx.drawImage(video, 0, 0, width, height);

      const mimeType = format === "png" ? "image/png" : "image/jpeg";
      const q = format === "jpeg" ? quality / 100 : undefined;
      const dataUrl = canvas.toDataURL(mimeType, q);

      frames.push({
        timestamp,
        dataUrl,
        width,
        height,
      });

      onProgress?.(i + 1, timestamps.length, `Extracted keyframe ${i + 1}/${timestamps.length}`);
    }

    video.src = "";
    video.load();

    return {
      frames,
      metadata: {
        ...metadata,
        width: video.videoWidth,
        height: video.videoHeight,
      },
      success: true,
    };
  } catch (error) {
    return {
      frames: [],
      metadata: { type: "video", sourceUrl: url },
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Create a thumbnail from the first frame of a video
 */
export async function createVideoThumbnail(
  url: string,
  maxWidth: number = 256,
  maxHeight: number = 256
): Promise<string | null> {
  try {
    const metadata = await getVideoMetadata(url);
    const duration = metadata.duration ?? 0;

    // Get frame at 10% of duration or 1 second, whichever is less
    const timestamp = Math.min(duration * 0.1, 1);

    const frame = await extractSingleFrame(url, timestamp, {
      format: "jpeg",
      quality: 70,
      scale: Math.min(maxWidth / (metadata.width ?? 640), maxHeight / (metadata.height ?? 480), 1),
    });

    return frame?.dataUrl ?? null;
  } catch {
    return null;
  }
}
