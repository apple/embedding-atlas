<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import PanelContainer from "./PanelContainer.svelte";
  import SchemaPanelRow from "./SchemaPanelRow.svelte";

  import { getStoreContext } from "../stores/embedding_atlas_store.js";

  const store = getStoreContext();

  const { columns, columnStyles } = store;
</script>

<PanelContainer title="Schema" class="p-2 flex flex-col gap-2">
  <div class="flex flex-col gap-2 overflow-x-hidden overflow-y-scroll">
    {#each $columns as column}
      <SchemaPanelRow
        column={column}
        style={$columnStyles[column.name] ?? {}}
        onChange={(s) => {
          store.setColumnStyle(column.name, s);
        }}
      />
    {/each}
  </div>
</PanelContainer>
