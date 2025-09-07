<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { ConfigContext } from "../../../context/config.svelte";

  interface Props {
    text: string | null;
    height: number;
    clamped: boolean;
    parentHeight: number;
  }

  let { text, height = $bindable(), clamped, parentHeight }: Props = $props();

  const config = ConfigContext.config;

  let element: HTMLElement | null = $state(null);
  let lines: number | null = $state(null);

  $effect(() => {
    // need to grab the scroll height because the text will be clamped
    if (element) {
      height = element.scrollHeight;
      lines = Math.floor(parentHeight / config.lineHeight);
    }
  });
</script>

<div class="text-content {clamped ? 'clamped' : null}" bind:this={element} style:--lines={lines}>
  {text}
</div>

<style>
  .clamped {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    line-clamp: var(--lines, var(--num-lines)); /* fallback to numlines from parent */
    -webkit-line-clamp: var(--lines, var(--num-lines));
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
