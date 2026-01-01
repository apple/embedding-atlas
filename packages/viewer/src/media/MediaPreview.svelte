<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import {
    detectMediaType,
    isYouTubeUrl,
    isVimeoUrl,
    extractYouTubeId,
    extractVimeoId,
    getYouTubeEmbedUrl,
    getYouTubeThumbnail,
    isDataUrl,
    isHttpUrl,
    type MediaType,
  } from "./types.js";

  interface Props {
    /** The source URL or data URL */
    src: string | null | undefined;
    /** Alternative text for accessibility */
    alt?: string;
    /** Maximum width in pixels */
    maxWidth?: number;
    /** Maximum height in pixels */
    maxHeight?: number;
    /** Whether to show controls for video */
    controls?: boolean;
    /** Whether to autoplay video (muted) */
    autoplay?: boolean;
    /** Whether to loop video */
    loop?: boolean;
    /** CSS class for the container */
    class?: string;
    /** Callback when media is loaded */
    onLoad?: () => void;
    /** Callback when media fails to load */
    onError?: (error: string) => void;
  }

  let {
    src,
    alt = "Media content",
    maxWidth = 800,
    maxHeight = 600,
    controls = true,
    autoplay = false,
    loop = false,
    class: className = "",
    onLoad,
    onError,
  }: Props = $props();

  let loading = $state(true);
  let error = $state<string | null>(null);
  let mediaType = $derived<MediaType>(src ? detectMediaType(src) : "unknown");

  // YouTube/Vimeo specific
  let youtubeId = $derived(src && isYouTubeUrl(src) ? extractYouTubeId(src) : null);
  let vimeoId = $derived(src && isVimeoUrl(src) ? extractVimeoId(src) : null);

  function handleLoad() {
    loading = false;
    error = null;
    onLoad?.();
  }

  function handleError(message: string) {
    loading = false;
    error = message;
    onError?.(message);
  }

  // Computed container style
  let containerStyle = $derived(
    `max-width: ${maxWidth}px; max-height: ${maxHeight}px;`
  );
</script>

<div
  class="media-preview {className}"
  style={containerStyle}
>
  {#if !src}
    <div class="placeholder">
      <span class="text-slate-400 dark:text-slate-600">No media</span>
    </div>
  {:else if loading && !youtubeId && !vimeoId}
    <div class="loading">
      <div class="spinner"></div>
      <span>Loading...</span>
    </div>
  {/if}

  {#if error}
    <div class="error">
      <span class="text-red-500">Failed to load: {error}</span>
    </div>
  {:else if youtubeId}
    <!-- YouTube embed -->
    <div class="video-container">
      <iframe
        src="{getYouTubeEmbedUrl(youtubeId)}?autoplay={autoplay ? 1 : 0}&loop={loop ? 1 : 0}&mute={autoplay ? 1 : 0}"
        title={alt}
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen
        onload={() => handleLoad()}
        onerror={() => handleError("Failed to load YouTube video")}
      ></iframe>
    </div>
  {:else if vimeoId}
    <!-- Vimeo embed -->
    <div class="video-container">
      <iframe
        src="https://player.vimeo.com/video/{vimeoId}?autoplay={autoplay ? 1 : 0}&loop={loop ? 1 : 0}&muted={autoplay ? 1 : 0}"
        title={alt}
        frameborder="0"
        allow="autoplay; fullscreen; picture-in-picture"
        allowfullscreen
        onload={() => handleLoad()}
        onerror={() => handleError("Failed to load Vimeo video")}
      ></iframe>
    </div>
  {:else if mediaType === "image"}
    <!-- Image -->
    <img
      {src}
      {alt}
      class="media-image"
      class:hidden={loading}
      onload={() => handleLoad()}
      onerror={() => handleError("Failed to load image")}
    />
  {:else if mediaType === "video"}
    <!-- Direct video -->
    <video
      {src}
      {controls}
      {autoplay}
      {loop}
      muted={autoplay}
      playsinline
      class="media-video"
      class:hidden={loading}
      onloadedmetadata={() => handleLoad()}
      onerror={() => handleError("Failed to load video")}
    >
      <track kind="captions" />
    </video>
  {:else if src && (isDataUrl(src) || isHttpUrl(src))}
    <!-- Unknown type but valid URL - try as image -->
    <img
      {src}
      {alt}
      class="media-image"
      class:hidden={loading}
      onload={() => handleLoad()}
      onerror={() => handleError("Failed to load content")}
    />
  {:else if src}
    <!-- Text content -->
    <div class="text-content">
      <p class="text-slate-700 dark:text-slate-300 whitespace-pre-wrap">{src}</p>
    </div>
  {/if}
</div>

<style>
  .media-preview {
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 200px;
    background-color: var(--bg-color, #f8fafc);
    border-radius: 0.5rem;
    overflow: hidden;
  }

  :global(.dark) .media-preview {
    --bg-color: #1e293b;
  }

  .placeholder,
  .loading,
  .error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 2rem;
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid #e2e8f0;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .video-container {
    position: relative;
    width: 100%;
    padding-bottom: 56.25%; /* 16:9 aspect ratio */
    height: 0;
  }

  .video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: none;
  }

  .media-image,
  .media-video {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }

  .text-content {
    padding: 1rem;
    max-height: 100%;
    overflow-y: auto;
    width: 100%;
  }

  .hidden {
    display: none;
  }
</style>
