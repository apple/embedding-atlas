<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Select from "../widgets/Select.svelte";
  import Slider from "../widgets/Slider.svelte";
  import Input from "../widgets/Input.svelte";

  import {
    type VideoProcessingOptions,
    type EmbeddingAggregation,
    defaultVideoProcessingOptions,
  } from "./types.js";

  interface Props {
    options: VideoProcessingOptions;
    onOptionsChange: (options: VideoProcessingOptions) => void;
    showAdvanced?: boolean;
  }

  let {
    options = defaultVideoProcessingOptions,
    onOptionsChange,
    showAdvanced = false,
  }: Props = $props();

  let showAdvancedSettings = $state(showAdvanced);

  function updateOptions(updates: Partial<VideoProcessingOptions>) {
    onOptionsChange({ ...options, ...updates });
  }

  function updateFrameExtraction(updates: Partial<typeof options.frameExtraction>) {
    onOptionsChange({
      ...options,
      frameExtraction: { ...options.frameExtraction, ...updates },
    });
  }

  const aggregationOptions: { value: EmbeddingAggregation; label: string }[] = [
    { value: "mean", label: "Mean Pooling (Average all frames)" },
    { value: "max", label: "Max Pooling (Maximum per dimension)" },
    { value: "temporal_pooling", label: "Temporal Pooling (Weight middle frames)" },
    { value: "attention_weighted", label: "Attention Weighted (Experimental)" },
  ];

  const formatOptions = [
    { value: "jpeg", label: "JPEG (Smaller, Lossy)" },
    { value: "png", label: "PNG (Larger, Lossless)" },
  ];
</script>

<div class="video-processing-settings space-y-4">
  <h4 class="text-sm font-semibold text-slate-700 dark:text-slate-300">
    Video Processing Settings
  </h4>

  <!-- Frame Rate -->
  <div class="space-y-1">
    <label class="block text-sm text-slate-600 dark:text-slate-400">
      Frames per Second: {options.frameExtraction.fps.toFixed(1)}
    </label>
    <div class="flex items-center gap-2">
      <Slider
        bind:value={
          () => options.frameExtraction.fps,
          (v) => updateFrameExtraction({ fps: v })
        }
        min={0.1}
        max={5}
        step={0.1}
      />
      <span class="text-xs text-slate-500 w-16">fps</span>
    </div>
    <p class="text-xs text-slate-500 dark:text-slate-400">
      Lower values = fewer frames, faster processing
    </p>
  </div>

  <!-- Max Frames -->
  <div class="space-y-1">
    <label class="block text-sm text-slate-600 dark:text-slate-400">
      Maximum Frames: {options.frameExtraction.maxFrames}
    </label>
    <div class="flex items-center gap-2">
      <Slider
        bind:value={
          () => options.frameExtraction.maxFrames,
          (v) => updateFrameExtraction({ maxFrames: Math.round(v) })
        }
        min={5}
        max={200}
        step={5}
      />
    </div>
    <p class="text-xs text-slate-500 dark:text-slate-400">
      Limits total frames extracted per video
    </p>
  </div>

  <!-- Aggregation Method -->
  <div class="space-y-1">
    <label class="block text-sm text-slate-600 dark:text-slate-400">
      Embedding Aggregation
    </label>
    <Select
      options={aggregationOptions}
      value={options.aggregation}
      onChange={(v) => updateOptions({ aggregation: v as EmbeddingAggregation })}
    />
    <p class="text-xs text-slate-500 dark:text-slate-400">
      How to combine frame embeddings into a single video embedding
    </p>
  </div>

  <!-- Store Frame Embeddings -->
  <label class="flex items-center gap-2 cursor-pointer">
    <input
      type="checkbox"
      checked={options.storeFrameEmbeddings}
      onchange={(e) => updateOptions({ storeFrameEmbeddings: e.currentTarget.checked })}
      class="w-4 h-4 rounded border-slate-300 dark:border-slate-600 text-blue-600 focus:ring-blue-500"
    />
    <span class="text-sm text-slate-600 dark:text-slate-400">
      Store individual frame embeddings
    </span>
  </label>
  <p class="text-xs text-slate-500 dark:text-slate-400 ml-6">
    Enables frame-level coding (increases storage)
  </p>

  <!-- Advanced Settings Toggle -->
  <button
    onclick={() => (showAdvancedSettings = !showAdvancedSettings)}
    class="text-sm text-blue-600 dark:text-blue-400 hover:underline"
  >
    {showAdvancedSettings ? "Hide" : "Show"} advanced settings
  </button>

  {#if showAdvancedSettings}
    <div class="space-y-4 pl-4 border-l-2 border-slate-200 dark:border-slate-700">
      <!-- Start Time -->
      <div class="space-y-1">
        <label class="block text-sm text-slate-600 dark:text-slate-400">
          Start Time (seconds)
        </label>
        <Input
          type="number"
          value={String(options.frameExtraction.startTime)}
          onChange={(e) => updateFrameExtraction({ startTime: parseFloat(e.currentTarget.value) || 0 })}
          className="w-32"
        />
      </div>

      <!-- End Time -->
      <div class="space-y-1">
        <label class="block text-sm text-slate-600 dark:text-slate-400">
          End Time (seconds, empty = full video)
        </label>
        <Input
          type="number"
          value={options.frameExtraction.endTime?.toString() ?? ""}
          onChange={(e) => {
            const val = parseFloat(e.currentTarget.value);
            updateFrameExtraction({ endTime: isNaN(val) ? undefined : val });
          }}
          className="w-32"
        />
      </div>

      <!-- Image Format -->
      <div class="space-y-1">
        <label class="block text-sm text-slate-600 dark:text-slate-400">
          Frame Image Format
        </label>
        <Select
          options={formatOptions}
          value={options.frameExtraction.format}
          onChange={(v) => updateFrameExtraction({ format: v as "jpeg" | "png" })}
        />
      </div>

      <!-- JPEG Quality -->
      {#if options.frameExtraction.format === "jpeg"}
        <div class="space-y-1">
          <label class="block text-sm text-slate-600 dark:text-slate-400">
            JPEG Quality: {options.frameExtraction.quality}%
          </label>
          <Slider
            bind:value={
              () => options.frameExtraction.quality,
              (v) => updateFrameExtraction({ quality: Math.round(v) })
            }
            min={10}
            max={100}
            step={5}
          />
        </div>
      {/if}

      <!-- Scale -->
      <div class="space-y-1">
        <label class="block text-sm text-slate-600 dark:text-slate-400">
          Scale Factor: {(options.frameExtraction.scale * 100).toFixed(0)}%
        </label>
        <Slider
          bind:value={
            () => options.frameExtraction.scale,
            (v) => updateFrameExtraction({ scale: v })
          }
          min={0.25}
          max={1}
          step={0.05}
        />
        <p class="text-xs text-slate-500 dark:text-slate-400">
          Reduce frame size for faster processing
        </p>
      </div>
    </div>
  {/if}
</div>

<style>
  .video-processing-settings {
    max-width: 400px;
  }
</style>
