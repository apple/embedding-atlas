<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import ContentRenderer from "../renderers/ContentRenderer.svelte";

  import { stringify, type ColumnStyle } from "../renderers/index.js";

  interface Props {
    values: Record<string, any>;
    columnStyles: Record<string, ColumnStyle>;
  }

  let { values, columnStyles }: Props = $props();

  let fullSizedKeys = $derived(Object.keys(columnStyles).filter((x) => columnStyles[x].display == "full"));
  let minifiedKeys = $derived(Object.keys(columnStyles).filter((x) => columnStyles[x].display == "badge"));
</script>

<div class="flex flex-col gap-2">
  <!-- Full sized fields -->
  {#each fullSizedKeys as key}
    {@const value = values[key]}

    <div class="flex flex-col">
      <div class="text-slate-400 dark:text-slate-400 font-medium text-xs">{key}</div>
      <div>
        <ContentRenderer
          value={value}
          renderer={columnStyles[key].renderer}
          rendererOptions={columnStyles[key].rendererOptions}
        />
      </div>
    </div>
  {/each}

  <!-- Minified fields -->
  <div class="flex-none flex flex-row gap-1 flex-wrap items-start">
    {#each minifiedKeys as key}
      {@const value = values[key]}

      <div
        class="px-2 flex items-center gap-2 border border-slate-200 dark:border-slate-700 bg-slate-100/25 dark:bg-slate-700/25 text-slate-700 dark:text-slate-300 rounded-md"
      >
        <div class="text-slate-400 dark:text-slate-400 font-medium text-sm">{key}</div>
        <div class="text-ellipsis whitespace-nowrap overflow-hidden max-w-72" title={stringify(value)}>
          <ContentRenderer value={value} />
        </div>
      </div>
    {/each}
  </div>
</div>
