<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script module lang="ts">
  export interface ListLayoutState {
    showTable?: boolean;
    showEmbedding?: boolean;
    showCharts?: boolean;

    chartsOrder?: string[];
    chartVisibility?: Record<string, boolean>;
  }

  export type Section = "embedding" | "table" | "chart";

  export function findSection(spec: any): Section | undefined {
    switch (spec.type) {
      case "embedding":
        return "embedding";
      case "table":
        return "table";
      default:
        return "chart";
    }
  }

  export function getSections(charts: Record<string, any>): Record<Section, string[]> {
    let r: Record<Section, string[]> = {
      embedding: [],
      table: [],
      chart: [],
    };
    for (let id in charts) {
      let section = findSection(charts[id]);
      if (section != undefined) {
        r[section].push(id);
      }
    }
    return r;
  }
</script>

<script lang="ts">
  import { deepMemo } from "@embedding-atlas/utils";
  import { flip } from "svelte/animate";
  import { slide } from "svelte/transition";

  import ListChartPanel from "./ListChartPanel.svelte";
  import Resizer from "./Resizer.svelte";

  import { findUnusedId } from "../../utils/identifier.js";
  import { reorder } from "../../utils/sort.js";
  import type { LayoutProps } from "../layout.js";

  let {
    context,
    charts,
    chartView,
    state: layoutState,
    onStateChange,
    onChartsChange,
    onChartStatesChange,
  }: LayoutProps<ListLayoutState> = $props();

  let { colorScheme } = context;

  let containerWidth = $state(100);
  let containerHeight = $state(100);

  let tableHeight = $state(300);
  let panelWidth = $state(400);
  let panelContainerWidth = $state(400);

  let sections = $derived.by(deepMemo(() => getSections(charts)));

  let isMobileLayout = $derived(containerWidth < 500);

  let hasEmbedding = $derived(sections.embedding.length > 0 && (layoutState.showEmbedding ?? true));
  let hasTable = $derived(sections.table.length > 0 && (layoutState.showTable ?? true));
  let hasChart = $derived(layoutState.showCharts ?? true);

  function chartWidth(total: number, desiredWidth: number) {
    const gap = 7;
    let nApprox = Math.round((total + gap) / (desiredWidth + gap));
    let minDiff: number | undefined = undefined;
    let minWidth: number | undefined = undefined;
    for (let n = Math.max(1, nApprox - 1); n <= Math.max(1, nApprox + 1); n++) {
      let preciseWidth = (total - gap * (n - 1)) / n;
      let diff = Math.abs(preciseWidth - desiredWidth);
      if (minDiff == undefined || diff < minDiff) {
        minDiff = diff;
        minWidth = preciseWidth;
      }
    }
    return Math.floor((minWidth ?? 400) * 2) / 2; // Round to multiple of 0.5
  }

  let chartsOrder = $derived.by(deepMemo(() => reorder(sections.chart, layoutState.chartsOrder)));

  function reorderCharts(id: string, shift: number) {
    let newOrder = [...chartsOrder];
    let index = newOrder.indexOf(id);
    if (index == -1) {
      return;
    }
    let targetIndex = index + shift;
    if (targetIndex < 0 || targetIndex >= newOrder.length) {
      return;
    }
    [newOrder[index], newOrder[targetIndex]] = [newOrder[targetIndex], newOrder[index]];
    onStateChange({ chartsOrder: newOrder });
  }

  function removeChart(id: string) {
    onChartsChange({ [id]: undefined });
    onChartStatesChange({ [id]: undefined });
  }
</script>

<div class="w-full h-full flex flex-row" bind:clientWidth={containerWidth} bind:clientHeight={containerHeight}>
  {#if !isMobileLayout}
    <!-- Desktop layout -->
    <!-- Left side: embedding / table -->
    {#if hasEmbedding || hasTable}
      <div class="flex-1 flex flex-col overflow-hidden">
        {#if hasEmbedding}
          <div class="flex flex-row gap-2 flex-1 overflow-hidden">
            {#each sections.embedding as id (id)}
              <div class="flex-1 overflow-hidden rounded-md">
                {@render chartView({ id: id, width: "container", height: "container" })}
              </div>
            {/each}
          </div>
        {/if}
        {#if hasEmbedding && hasTable}
          <Resizer
            class="h-2 flex-none"
            axis="y"
            min={100}
            max={containerHeight - 100}
            scaler={-1}
            value={tableHeight}
            onChange={(v) => (tableHeight = v)}
          />
        {/if}
        {#if hasTable}
          <div
            class={hasEmbedding ? "flex-none" : "flex-1"}
            style:height={hasEmbedding ? `${tableHeight}px` : null}
            transition:slide
          >
            {#each sections.table as id (id)}
              <div class="flex-1 h-full overflow-hidden rounded-md">
                {@render chartView({ id: id, width: "container", height: "container" })}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
    {#if (hasEmbedding || hasTable) && hasChart}
      <Resizer
        class="w-2 flex-none"
        axis="x"
        min={100}
        max={containerWidth - 100}
        scaler={-1}
        value={panelWidth}
        onChange={(v) => (panelWidth = v)}
      />
    {/if}
    <!-- Right side: charts -->
    {#if hasChart}
      <div
        class="h-full overflow-x-hidden overflow-y-scroll"
        style:width="{hasEmbedding || hasTable ? panelWidth : containerWidth}px"
        transition:slide={{ axis: "x" }}
      >
        <div class="flex flex-row flex-wrap gap-2" bind:clientWidth={panelContainerWidth}>
          <button
            class="bg-white dark:bg-black rounded-md flex flex-col justify-center items-center gap-2 p-2 w-full text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 select-none"
            onclick={() => {
              let id = findUnusedId(charts);
              onChartsChange({ [id]: { type: "builder", title: "New" } });
              onStateChange({ chartsOrder: [id, ...chartsOrder.filter((x) => x != id)] });
            }}
          >
            + Add
          </button>
          {#each chartsOrder as id, index (id)}
            {@const spec = charts[id]}
            {@const isVisible = layoutState.chartVisibility?.[id] ?? true}
            <div
              class="bg-white dark:bg-black rounded-md flex flex-col group"
              style:width="{chartWidth(panelContainerWidth, 500)}px"
              animate:flip={{ duration: 300 }}
              out:slide
            >
              <ListChartPanel
                id={id}
                spec={spec}
                onIsVisibleChange={(v) => {
                  onStateChange({ chartVisibility: { [id]: v } });
                }}
                isVisible={isVisible}
                colorScheme={$colorScheme}
                chartView={chartView}
                onRemove={removeChart.bind(null, id)}
                onUp={index > 0 ? reorderCharts.bind(null, id, -1) : undefined}
                onDown={index + 1 < chartsOrder.length ? reorderCharts.bind(null, id, 1) : undefined}
                onSpecChange={(spec) => {
                  onChartsChange({ [id]: undefined });
                  onChartStatesChange({ [id]: undefined });
                  onChartsChange({ [id]: spec });
                }}
              />
            </div>
          {/each}
        </div>
      </div>
    {/if}
  {:else}
    <!-- Mobile layout -->
    <div class="w-full h-full overflow-y-scroll flex flex-col gap-2">
      {#each sections.embedding.concat(chartsOrder, sections.table) as id, index (id)}
        {@const isVisible = layoutState.chartVisibility?.[id] ?? true}
        {@const indexInCharts = chartsOrder.indexOf(id)}
        <div class="bg-white dark:bg-black rounded-md flex flex-col group" animate:flip={{ duration: 300 }} out:slide>
          <ListChartPanel
            id={id}
            spec={charts[id]}
            onIsVisibleChange={(v) => {
              onStateChange({ chartVisibility: { [id]: v } });
            }}
            isVisible={isVisible}
            colorScheme={$colorScheme}
            chartView={chartView}
            onRemove={removeChart.bind(null, id)}
            onUp={indexInCharts > 0 ? reorderCharts.bind(null, id, -1) : undefined}
            onDown={indexInCharts != -1 && indexInCharts + 1 < chartsOrder.length
              ? reorderCharts.bind(null, id, 1)
              : undefined}
          />
        </div>
      {/each}
    </div>
  {/if}
</div>
