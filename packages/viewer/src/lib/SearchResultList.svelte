<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Mark from "mark.js";

  import TooltipContent from "./TooltipContent.svelte";

  import { IconClose } from "./icons.js";
  import type { ColumnStyle } from "./renderers/index.js";
  import type { SearchResultItem } from "./search.js";

  interface Props {
    items: SearchResultItem[];
    label: string;
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

<div class="flex flex-col w-full h-full">
  <div class="ml-3 mr-2 my-1 flex items-center text-slate-400 dark:text-slate-500 items-start">
    <div class="flex-1">
      <div>{label}</div>
      <div>{resultCountText}</div>
    </div>
    <div class="flex-none mt-1">
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
  <hr class="border-slate-300 dark:border-slate-600" />
  <div class="flex flex-col overflow-x-hidden overflow-y-scroll">
    {#each items as item (item)}
      <button
        class="m-1 p-2 text-left rounded-md hover:outline outline-slate-500"
        onclick={() => {
          onClick?.(item);
        }}
      >
        {#if item.distance != null}
          <div class="flex pb-1 text-sm">
            <span class="px-2 flex gap-2 bg-slate-200 text-slate-500 dark:bg-slate-600 dark:text-slate-300 rounded-md">
              <div class="text-slate-400 dark:text-slate-400 font-medium">Distance</div>
              <div class="text-ellipsis whitespace-nowrap overflow-hidden max-w-72">
                {item.distance.toFixed(5)}
              </div>
            </span>
          </div>
        {/if}
        <div class="overflow-hidden text-ellipsis line-clamp-4 leading-5" use:markHighlight={highlight}>
          <TooltipContent values={item.fields} columnStyles={columnStyles ?? {}} />
        </div>
      </button>
      <hr class="border-slate-300 dark:border-slate-600" />
    {/each}
  </div>
</div>
