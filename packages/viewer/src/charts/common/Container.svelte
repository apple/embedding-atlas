<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    width?: number;
    height?: number;
    defaultWidth?: number | null;
    defaultHeight?: number | null;
    scrollX?: boolean;
    scrollY?: boolean;
    class?: string;
    children: Snippet;
  }

  let {
    width,
    height,
    defaultWidth = null,
    defaultHeight = null,
    scrollX = false,
    scrollY = false,
    class: className,
    children,
  }: Props = $props();

  let resolvedWidth = $derived(width != null ? width : scrollX ? undefined : defaultWidth);
  let resolvedHeight = $derived(height != null ? height : scrollY ? undefined : defaultHeight);
</script>

<div
  class="p-2 {className ?? ''}"
  style:width={resolvedWidth != null ? `${resolvedWidth}px` : undefined}
  style:height={resolvedHeight != null ? `${resolvedHeight}px` : undefined}
  style:overflow-x={scrollX && width != null ? "auto" : undefined}
  style:overflow-y={scrollY && height != null ? "auto" : undefined}
>
  {@render children()}
</div>
