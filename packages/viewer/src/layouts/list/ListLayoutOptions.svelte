<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import ToggleButton from "../../widgets/ToggleButton.svelte";
  import { getSections } from "./ListLayout.svelte";

  import { IconEmbeddingView, IconMenu, IconTable } from "../../assets/icons.js";

  import { getStoreContext } from "../../stores/embedding_atlas_store.js";
  import type { LayoutOptionsProps } from "../layout.js";
  import type { ListLayoutSpec } from "./types.js";

  let { layout }: LayoutOptionsProps = $props();

  const store = getStoreContext();
  const { charts, layouts } = store;
  let spec = $derived($layouts[layout]) as ListLayoutSpec;

  let sections = $derived(getSections($charts, spec.chartIds));
</script>

<div class="flex gap-0.5 items-center">
  {#if sections.embedding.length > 0}
    <ToggleButton
      icon={IconEmbeddingView}
      title="Show / hide embedding"
      bind:checked={
        () => spec.showEmbedding ?? true,
        (v) => {
          store.updateLayout<ListLayoutSpec>(layout, (draft) => {
            draft.showEmbedding = v;
          });
        }
      }
    />
  {/if}
  {#if sections.table.length > 0}
    <ToggleButton
      icon={IconTable}
      title="Show / hide table"
      bind:checked={
        () => spec.showTable ?? true,
        (v) => {
          store.updateLayout<ListLayoutSpec>(layout, (draft) => {
            draft.showTable = v;
          });
        }
      }
    />
  {/if}
  <ToggleButton
    icon={IconMenu}
    title="Show / hide charts"
    bind:checked={
      () => spec.showCharts ?? true,
      (v) => {
        store.updateLayout<ListLayoutSpec>(layout, (draft) => {
          draft.showCharts = v;
        });
      }
    }
  />
</div>
