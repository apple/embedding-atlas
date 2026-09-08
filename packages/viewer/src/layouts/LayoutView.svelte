<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import ChartView from "../charts/ChartView.svelte";

  import { getStoreContext } from "../stores/embedding_atlas_store.js";
  import { findLayoutComponent } from "./layout_types.js";
  import NewTabView from "./NewTabView.svelte";

  const store = getStoreContext();
  const { charts, chartStates, layouts, currentLayout, chartContext } = store;

  let spec = $derived($layouts[$currentLayout] as any);
  let type = $derived(spec?.type);
  let LayoutClass = $derived(type != undefined ? findLayoutComponent(type) : undefined);
</script>

{#if LayoutClass != undefined}
  <LayoutClass layout={$currentLayout}>
    {#snippet chartView({ id, width, height, mode })}
      {@const spec = $charts[id]}
      {@const state = $chartStates[id]}
      <ChartView
        context={chartContext}
        width={width}
        height={height}
        spec={spec}
        state={state}
        mode={mode ?? "view"}
        onSpecChange={store.updateChart.bind(store, id)}
        onStateChange={store.updateChartState.bind(store, id)}
        registerDelegate={store.registerChartDelegate.bind(store, id)}
      />
    {/snippet}
  </LayoutClass>
{:else}
  <div class="justify-center items-center flex h-full w-full bg-slate-100 dark:bg-slate-900 rounded-md">
    <NewTabView />
  </div>
{/if}
