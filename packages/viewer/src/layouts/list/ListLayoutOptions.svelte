<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import ToggleButton from "../../widgets/ToggleButton.svelte";
  import { getSections } from "./ListLayout.svelte";

  import { IconEmbeddingView, IconMenu, IconTable } from "../../assets/icons.js";

  import type { LayoutOptionsProps } from "../layout.js";
  import type { ListLayoutState } from "./types.js";

  let { charts, state, onStateChange }: LayoutOptionsProps<ListLayoutState> = $props();

  let sections = $derived(getSections(charts));
</script>

<div class="flex gap-0.5 items-center">
  {#if sections.embedding.length > 0}
    <ToggleButton
      icon={IconEmbeddingView}
      title="Show / hide embedding"
      bind:checked={
        () => state.showEmbedding ?? true,
        (v) => {
          onStateChange({ showEmbedding: v });
        }
      }
    />
  {/if}
  <ToggleButton
    icon={IconTable}
    title="Show / hide table"
    bind:checked={
      () => state.showTable ?? true,
      (v) => {
        onStateChange({ showTable: v });
      }
    }
  />
  <ToggleButton
    icon={IconMenu}
    title="Show / hide charts"
    bind:checked={
      () => state.showCharts ?? true,
      (v) => {
        onStateChange({ showCharts: v });
      }
    }
  />
</div>
