<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Input from "../widgets/Input.svelte";
  import Select from "../widgets/Select.svelte";
  import Spinner from "../widgets/Spinner.svelte";
  import PanelContainer from "./PanelContainer.svelte";
  import SearchResultList from "./SearchResultList.svelte";

  import { getStoreContext } from "../stores/embedding_atlas_store.js";

  const store = getStoreContext();
  const { search, columns, columnStyles, chartContext } = store;

  const { result, status, query, mode, searchModes, textColumn } = search;
</script>

<PanelContainer title="Search" class="p-2 flex flex-col gap-2">
  <div class="flex gap-2">
    <Select
      options={[
        { value: "full-text", label: "Full Text" },
        { value: "vector", label: "Vector" },
      ]}
      value={$mode}
      onChange={(v) => ($mode = v)}
    />
    {#if $mode == "full-text"}
      <Select
        class="flex-1"
        options={[
          ...($columns.filter((x) => x.jsType == "string").length == 0 ? [{ value: undefined, label: "(none)" }] : []),
          ...$columns
            .filter((x) => x.jsType == "string")
            .map((c) => ({ value: c.name, label: c.name + " (" + c.type + ")" })),
        ]}
        value={$textColumn}
        onChange={(v) => ($textColumn = v)}
      />
    {/if}
  </div>

  {#if $searchModes.indexOf($mode) >= 0}
    <Input type="search" className="w-full" bind:value={$query} />
  {:else}
    <div class="text-slate-500 dark:text-slate-400">
      {#if $mode == "full-text"}Select a text column to enable search.{/if}
      {#if $mode == "vector"}Vector search is not supported by the backend.{/if}
    </div>
  {/if}

  {#if $status != null}
    <Spinner status={$status} />
  {/if}

  {#if $result != null}
    {@const searchResult = $result}
    {#key searchResult}
      <SearchResultList
        items={searchResult.items}
        label={searchResult.label}
        highlight={searchResult.highlight}
        limit={searchResult.limit}
        onClick={async (item) => {
          chartContext.highlight.set([item.id]);
        }}
        onClose={() => search.clear()}
        columnStyles={$columnStyles}
      />
    {/key}
  {/if}
</PanelContainer>
