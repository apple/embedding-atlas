<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { coordinator as defaultCoordinator, wasmConnector } from "@uwdata/mosaic-core";

  import { Table } from "embedding-atlas/svelte";
  import { createSampleDataTable } from "../sample_datasets.js";

  const coordinator = defaultCoordinator();

  let columns: string[] = [];

  let initialized = (async () => {
    const wasm = await wasmConnector();
    coordinator.databaseConnector(wasm);
    await createSampleDataTable(coordinator, "dataset", 100000);
    let describe = await coordinator.query("DESCRIBE dataset");
    columns = Array.from(describe.getChild("column_name"));
  })();
</script>

{#await initialized}
  Initializing dataset...
{:then}
  <div class="w-full h-full">
    <Table table="dataset" columns={columns} rowKey="id" />
  </div>
{/await}
