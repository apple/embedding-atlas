<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Slider from "../widgets/Slider.svelte";
  import type { RendererOptionsProps } from "./types.js";

  let { options, onChange }: RendererOptionsProps = $props();
</script>

<div class="flex gap-2 items-center">
  <div class="text-slate-500 dark:text-slate-400 select-none">Size</div>
  <Slider
    bind:value={
      () => options?.size ?? 100,
      (v) => {
        onChange?.({ size: v });
      }
    }
    width={80}
    min={16}
    max={400}
    step={1}
  />
</div>
