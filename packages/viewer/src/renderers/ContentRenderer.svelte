<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { imageToDataUrl } from "@embedding-atlas/utils";

  import AudioContent from "./AudioContent.svelte";

  import { renderers } from "./renderer_types.js";
  import { stringify, valueKind } from "./renderer_utils.js";
  import type { ColumnStyle } from "./types.js";

  interface Props {
    value?: string;
    style?: ColumnStyle;
    formatter?: (value: any) => string;
  }

  let { value = "", style, formatter }: Props = $props();

  let renderer = $derived(style?.renderer);
  let rendererOptions = $derived(style?.options ?? {});
  let kind = $derived(valueKind(value));

  let rendererAction = $derived(renderer != null ? (renderers[renderer] ?? null) : null);

  function stringifyWithFormatter(value: any, formatter?: (value: any) => string) {
    if (formatter && typeof value == "number") {
      return formatter(value);
    } else if (formatter && value instanceof Date) {
      return formatter(value.getTime());
    } else {
      return stringify(value);
    }
  }
</script>

{#if rendererAction == null}
  {#if kind == "link"}
    <a href={value} class="underline" target="_blank" rel="noopener noreferrer">{value}</a>
  {:else if kind == "image"}
    <img src={imageToDataUrl(value)} alt="" referrerpolicy="no-referrer" class="max-w-24 max-h-24" />
  {:else if kind == "audio"}
    <AudioContent value={value} />
  {:else}
    {stringifyWithFormatter(value, formatter)}
  {/if}
{:else}
  {#key rendererAction}
    <div use:rendererAction={{ value: value, options: rendererOptions }}></div>
  {/key}
{/if}
