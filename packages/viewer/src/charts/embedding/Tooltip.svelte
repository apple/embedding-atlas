<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { DataPoint } from "@embedding-atlas/component";

  import TooltipContent from "../../views/TooltipContent.svelte";

  import { IconSearch } from "../../assets/icons.js";
  import type { ColumnStyle } from "../../renderers/index.js";

  interface Props {
    tooltip: DataPoint;
    columnStyles?: Record<string, ColumnStyle>;
    colorScheme: "light" | "dark";
    onNearestNeighborSearch?: (id: any) => void;
  }

  let { tooltip, columnStyles, colorScheme, onNearestNeighborSearch }: Props = $props();
</script>

<div class="embedding-atlas-root">
  <div
    class="p-2 border flex flex-col gap-2 border-slate-300 dark:border-slate-600 shadow-md text-slate-700 dark:text-slate-300 rounded-md text-ellipsis overflow-x-hidden overflow-y-scroll bg-white/75 dark:bg-slate-800/75 backdrop-blur-sm"
    class:dark={colorScheme == "dark"}
    style:max-width="400px"
    style:max-height="300px"
  >
    <TooltipContent values={tooltip.fields ?? {}} columnStyles={columnStyles ?? {}} />
    {#if onNearestNeighborSearch}
      <div>
        <button
          class="text-sm flex gap-0.5 items-center text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-300"
          onclick={() => {
            onNearestNeighborSearch?.(tooltip.identifier);
          }}
        >
          <IconSearch /> Nearest Neighbors
        </button>
      </div>
    {/if}
  </div>
</div>
