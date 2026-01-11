<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    url: string | null;
    height: number;
  }

  let { url, height = $bindable() }: Props = $props();
</script>

<div class="link-content" bind:clientHeight={height}>
  {#if url}
    <a href={url} target="_blank" rel="noopener noreferrer">{url}</a>
  {/if}
</div>

<style>
</style>
