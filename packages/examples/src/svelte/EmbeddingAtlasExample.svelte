<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";

  import { EmbeddingAtlas } from "embedding-atlas/svelte";
  import { createSampleDataTable } from "../sample_datasets.js";

  const coordinator = new Coordinator();

  let initialized = (async () => {
    const wasm = await wasmConnector();
    coordinator.databaseConnector(wasm);
    await createSampleDataTable(coordinator, "dataset", 100000);
  })();
</script>

{#await initialized}
  Initializing dataset...
{:then}
  <div class="w-full h-full">
    <EmbeddingAtlas
      coordinator={coordinator}
      data={{
        table: "dataset",
        id: "id",
        text: "text",
        projection: { x: "x", y: "y" },
      }}
    />
  </div>
{/await}
