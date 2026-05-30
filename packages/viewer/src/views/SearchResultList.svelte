<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Mark from "mark.js";

  import Paginator from "../widgets/Paginator.svelte";
  import TooltipContent from "./TooltipContent.svelte";

  import { IconClose } from "../assets/icons.js";
  import type { ColumnStyle } from "../renderers/types.js";
  import type { SearchResultItem } from "../search/search.js";

  interface Props {
    items: SearchResultItem[];
    label?: string;
    highlight: string;
    limit?: number;
    columnStyles?: Record<string, ColumnStyle>;
    onClick?: (item: SearchResultItem) => void;
    onClose?: () => void;
  }

  let { items, label, highlight, limit = 100, columnStyles, onClick, onClose }: Props = $props();

  function markHighlight(element: HTMLElement, highlight: string) {
    let m = new Mark(element);
    m.mark(highlight);
  }

  let resultCountText = $derived(
    items.length == 0
      ? "No result found."
      : items.length == 1
        ? `${items.length.toLocaleString()} result.`
        : items.length >= limit
          ? `More than ${items.length.toLocaleString()} results, showing top ${limit.toLocaleString()}.`
          : `${items.length.toLocaleString()} results.`,
  );
</script>

<div class="flex-1 flex flex-col gap-2 w-full overflow-y-hidden">
  <div class="flex items-center items-start">
    <div class="flex-1 flex flex-col">
      <div class="text-slate-500 dark:text-slate-400">
        {label}
      </div>
      <div class="text-slate-400 dark:text-slate-500">
        {resultCountText}
      </div>
    </div>
    <div class="flex-none">
      <button
        class="block hover:text-slate-500 dark:hover:text-slate-400"
        onclick={() => {
          onClose?.();
        }}
      >
        <IconClose />
      </button>
    </div>
  </div>

  <Paginator count={items.length}>
    {#snippet children({ start, end })}
      <div class="flex flex-col gap-2 overflow-x-hidden overflow-y-scroll">
        {#each items.slice(start, end) as item (item)}
          <button
            class="p-2 text-left rounded-md hover:border-blue-400 border border-slate-200 dark:border-slate-700 bg-white dark:bg-black select-text"
            onclick={() => {
              onClick?.(item);
            }}
          >
            {#if item.distance != null}
              <div class="flex pb-1 text-sm">
                <span
                  class="px-2 flex gap-2 bg-slate-200 text-slate-500 dark:bg-slate-600 dark:text-slate-300 rounded-md"
                >
                  <div class="text-slate-400 dark:text-slate-400 font-medium">Distance</div>
                  <div class="text-ellipsis whitespace-nowrap overflow-hidden max-w-72">
                    {item.distance.toFixed(5)}
                  </div>
                </span>
              </div>
            {/if}
            <div use:markHighlight={highlight}>
              <TooltipContent values={item.fields} columnStyles={columnStyles ?? {}} />
            </div>
          </button>
        {/each}
      </div>
    {/snippet}
  </Paginator>
</div>
