<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";

  import { EmbeddingAtlas, registerRenderer } from "embedding-atlas/svelte";
  import { createSampleDataTable } from "../sample_datasets.js";

  const coordinator = new Coordinator();

  class TagsRenderer {
    node: HTMLElement;

    constructor(node: HTMLElement, props: { value: string }) {
      this.node = node;
      this.update(props);
    }

    update(props: { value: string }) {
      let el = document.createElement("span");
      el.innerText = props.value.toString();
      el.style = "border: 1px solid #ccc; border-radius: 2px; padding: 2px 4px";
      this.node.replaceChildren(el);
    }
  }

  registerRenderer({ name: "custom_renderer", label: "Custom Renderer", renderer: TagsRenderer });

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
      initialState={{
        columnStyles: {
          var_many_category: { renderer: "custom_renderer" },
        },
      }}
    />
  </div>
{/await}
