<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import ActionButton from "../widgets/ActionButton.svelte";
  import PopupButton from "../widgets/PopupButton.svelte";
  import Select from "../widgets/Select.svelte";
  import SelectionCount from "./SelectionCount.svelte";

  import { IconClose, IconDownload } from "../assets/icons.js";

  import { getStoreContext } from "../stores/embedding_atlas_store.js";

  const store = getStoreContext();
  const { coordinator, crossFilter, chartContext } = store;

  let exportFormat: "json" | "jsonl" | "csv" | "parquet" = $state("parquet");
</script>

<div
  class="flex flex-none gap-2 items-center pl-2 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900"
>
  <SelectionCount coordinator={coordinator} filter={crossFilter} table={chartContext.table} />
  <div class="flex flex-row items-center">
    <button
      title="Clear filters"
      onclick={() => store.resetFilter()}
      class="rounded-md flex select-none items-center p-1.5 text-slate-400 dark:text-slate-500 hover:text-slate-500 dark:hover:text-slate-400 focus-visible:outline-2 outline-blue-600 -outline-offset-1"
    >
      <IconClose class="w-5 h-5" />
    </button>

    {#if store.props.onExportSelection}
      {@const onExportSelection = store.props.onExportSelection}
      <PopupButton title="Export Selection">
        {#snippet button({ visible, toggle })}
          <button
            title="Export Selection"
            onclick={toggle}
            class="rounded-md px-1.5 py-1.5 flex select-none items-center text-slate-400 dark:text-slate-500 hover:text-slate-500 dark:hover:text-slate-400 focus-visible:outline-2 outline-blue-600 -outline-offset-1"
            class:text-slate-400={!visible}
            class:dark:text-slate-500={!visible}
          >
            <IconDownload class="w-5 h-5" />
          </button>
        {/snippet}
        <div class="flex flex-col gap-2 select-none">
          <div class="text-slate-500 dark:text-slate-400">Export the selected data points</div>
          <div class="flex gap-2">
            <ActionButton
              icon={IconDownload}
              label="Export Selection"
              title="Export the selected points"
              class="w-48"
              onClick={() => onExportSelection(store.getCurrentPredicate(), exportFormat)}
            />
            <Select
              label="Format"
              value={exportFormat}
              onChange={(v) => (exportFormat = v)}
              options={[
                { value: "parquet", label: "Parquet" },
                { value: "jsonl", label: "JSONL" },
                { value: "json", label: "JSON" },
                { value: "csv", label: "CSV" },
              ]}
            />
          </div>
        </div>
      </PopupButton>
    {/if}
  </div>
</div>
