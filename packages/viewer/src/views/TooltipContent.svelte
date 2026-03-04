<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import ContentRenderer from "../renderers/ContentRenderer.svelte";

  import { stringify } from "../renderers/renderer_utils.js";
  import { type ColumnStyle } from "../renderers/types.js";

  interface Props {
    values: Record<string, any>;
    columns?: string[];
    columnStyles: Record<string, ColumnStyle>;
  }

  let { columns, values, columnStyles }: Props = $props();

  function keyStyle(key: string, columnStyles: Record<string, ColumnStyle>): "full" | "badge" | "hidden" {
    let display = columnStyles[key]?.display;
    if (display == null) {
      if (key.startsWith("__")) {
        return "hidden";
      } else {
        return "full";
      }
    }
    return display;
  }

  let allKeys = $derived(columns ?? Object.keys(values));
  let fullKeys = $derived(allKeys.filter((k) => keyStyle(k, columnStyles) == "full"));
  let badgeKeys = $derived(allKeys.filter((k) => keyStyle(k, columnStyles) == "badge"));
</script>

<div class="flex flex-col gap-2">
  <!-- Full sized fields -->
  {#each fullKeys as key}
    {@const value = values[key]}

    <div class="flex flex-col">
      <div class="text-slate-400 dark:text-slate-400 font-medium text-xs">{key}</div>
      <div>
        <ContentRenderer value={value} style={columnStyles[key]} />
      </div>
    </div>
  {/each}

  <!-- Minified fields -->
  <div class="flex-none flex flex-row gap-1 flex-wrap items-start">
    {#each badgeKeys as key}
      {@const value = values[key]}

      <div
        class="px-2 flex items-center gap-2 border border-slate-200 dark:border-slate-700 bg-slate-100/25 dark:bg-slate-700/25 text-slate-700 dark:text-slate-300 rounded-md min-w-0"
      >
        <div class="text-slate-400 dark:text-slate-400 font-medium text-sm flex-shrink-0">{key}</div>
        <div class="text-ellipsis whitespace-nowrap overflow-hidden max-w-72 min-w-0" title={stringify(value)}>
          <ContentRenderer value={value} />
        </div>
      </div>
    {/each}
  </div>
</div>
