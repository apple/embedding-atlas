<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    flexHeight?: boolean;
    defaultHeight?: number;
    children?: Snippet<[number, number]>;
  }

  let containerWidth = $state(300);
  let containerHeight = $state(200);

  let { flexHeight = true, defaultHeight = 200, children }: Props = $props();
</script>

<div
  style:user-select="none"
  style:position="relative"
  style:height={!flexHeight ? `${defaultHeight}px` : undefined}
  class={flexHeight ? "flex-1 min-h-0" : undefined}
  bind:clientWidth={containerWidth}
  bind:clientHeight={containerHeight}
>
  {@render children?.(containerWidth, containerHeight)}
</div>
