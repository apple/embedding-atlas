<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { imageToDataUrl } from "../utils/image.js";
  import { renderers } from "./renderer_types.js";
  import { isImage, isLink, stringify } from "./renderer_utils.js";
  import type { ColumnStyle } from "./types.js";

  interface Props {
    value?: string;
    style?: ColumnStyle;
  }

  let { value = "", style }: Props = $props();

  let renderer = $derived(style?.renderer);
  let rendererOptions = $derived(style?.options ?? {});

  let rendererAction = $derived(renderer != null ? (renderers[renderer] ?? null) : null);
</script>

{#if rendererAction == null}
  {#if isLink(value)}
    <a href={value} class="underline" target="_blank" rel="noopener noreferrer">{value}</a>
  {:else if isImage(value)}
    <img src={imageToDataUrl(value)} alt="" referrerpolicy="no-referrer" class="max-w-24 max-h-24" />
  {:else}
    {stringify(value)}
  {/if}
{:else}
  {#key rendererAction}
    <div use:rendererAction={{ value: value, options: rendererOptions }}></div>
  {/key}
{/if}
