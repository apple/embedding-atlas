<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconClose, IconSortDown, IconSortUp } from "../../assets/icons.js";
  import type { SortOrder } from "./types.js";

  interface Props {
    value?: SortOrder;
    onChange?: (value: SortOrder | undefined) => void;
  }

  let { value, onChange }: Props = $props();

  function flipDirection(column: string) {
    onChange?.(
      value?.map((x) => {
        if (x.column == column) {
          return { column: x.column, direction: x.direction == "ascending" ? "descending" : "ascending" };
        } else {
          return x;
        }
      }),
    );
  }

  function clear(column: string) {
    let newValue = value?.filter((x) => x.column != column);
    if (newValue?.length == 0) {
      newValue = undefined;
    }
    onChange?.(newValue);
  }
</script>

{#if value?.length ?? 0 > 0}
  <div class="flex gap-2 items-center select-none">
    {#each value ?? [] as item, index}
      <div
        class="flex items-center bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded-md text-slate-500 dark:text-slate-400"
      >
        {item.column}
        <button onclick={() => flipDirection(item.column)}>
          {#if item.direction == "ascending"}
            <IconSortUp />
          {:else if item.direction == "descending"}
            <IconSortDown />
          {/if}
        </button>
        <button class="text-slate-500 dark:text-slate-400 text-sm" onclick={() => clear(item.column)}>
          <IconClose />
        </button>
      </div>
    {/each}
  </div>
{/if}
