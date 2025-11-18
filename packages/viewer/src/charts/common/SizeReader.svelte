<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    flexHeight?: boolean;

    children?: Snippet<[number, number]>;
  }

  let containerWidth = $state(300);
  let containerHeight = $state(200);

  let { flexHeight = false, children }: Props = $props();
</script>

<div
  style:user-select="none"
  style:position="relative"
  style:height={!flexHeight ? `fit-content` : undefined}
  class={flexHeight ? "flex-1 min-h-0" : undefined}
  bind:clientWidth={containerWidth}
  bind:clientHeight={containerHeight}
>
  {#if flexHeight}
    <div style:position="absolute" style:height="{containerHeight}px">
      {@render children?.(containerWidth, containerHeight)}
    </div>
  {:else}
    {@render children?.(containerWidth, containerHeight)}
  {/if}
</div>
