<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { untrack } from "svelte";
  import { fade } from "svelte/transition";

  import DashboardChartPanel from "./DashboardChartPanel.svelte";

  import { IconPlus } from "../../assets/icons.js";

  import { deepMemo } from "../../../../utils/dist/equals.js";
  import { findUnusedId } from "../../utils/identifier.js";
  import { reorder } from "../../utils/sort.js";
  import type { LayoutProps } from "../layout.js";
  import { OccupancyMap } from "./occupancy_map.js";
  import { Grid, computePlacements, type Placement } from "./placement.js";
  import type { DashboardLayoutState } from "./types.js";

  let {
    context,
    charts,
    state: layoutState,
    onStateChange,
    onChartsChange,
    onChartStatesChange,
    chartView,
  }: LayoutProps<DashboardLayoutState> = $props();

  let containerWidth = $state(100);
  let containerHeight = $state(100);

  let innerContainer: HTMLDivElement;

  let placementChangingCounter = $state(0);

  let numColumns = $derived(layoutState.numColumns ?? (containerWidth < 500 ? 8 : 24));
  let numRows = $derived(layoutState.numRows ?? 16);
  let grid = $derived(new Grid(containerWidth, containerHeight, numColumns, numRows, 6));
  let gridKey = $derived(`${numColumns}x${numRows}`);

  let placements = $derived.by(() =>
    computePlacements(charts, layoutState.grids?.[gridKey]?.placements ?? {}, grid.numColumns),
  );
  let chartIds = $derived.by(
    deepMemo(() => reorder(Object.keys(charts), layoutState.grids?.[gridKey]?.order).reverse()),
  );
  let newChartRects = $derived.by(() => {
    let map = new OccupancyMap(grid.numColumns);
    for (let id in placements) {
      map.fill(placements[id].x, placements[id].y, placements[id].width, placements[id].height);
    }
    return map.unusedRects(4, 4, 12, 8);
  });

  let maxY = $derived.by(() => {
    let maxY = grid.numRows;
    for (let id in placements) {
      maxY = Math.max(maxY, placements[id].y + placements[id].height);
    }
    return maxY;
  });

  let lockedMaxY = $state<number | undefined>(undefined);
  let effectiveMaxY = $derived(Math.max(lockedMaxY ?? maxY, maxY));
  let innerHeight = $derived(effectiveMaxY <= grid.numRows ? containerHeight : grid.yScaler * effectiveMaxY);

  $effect.pre(() => {
    let cond = placementChangingCounter >= 1;
    untrack(() => {
      if (cond) {
        lockedMaxY = Math.max(lockedMaxY ?? maxY, maxY);
      } else {
        lockedMaxY = 0;
      }
    });
  });

  function removeChart(id: string) {
    onChartsChange({ [id]: undefined });
    onChartStatesChange({ [id]: undefined });
  }

  function bringToFront(id: string) {
    let existingOrder = layoutState.grids?.[gridKey]?.order ?? [];
    let newOrder = [id, ...existingOrder.filter((x) => x != id)];
    onStateChange({ grids: { [gridKey]: { order: newOrder } } });
  }

  function newChart(placement: Placement) {
    let id = findUnusedId(charts);
    onChartsChange({ [id]: { type: "builder", title: "New" } });
    onStateChange({ grids: { [gridKey]: { placements: { [id]: placement } } } });
  }
</script>

<div
  class="w-full h-full overflow-x-hidden relative {innerHeight > containerHeight
    ? 'overflow-y-scroll overscroll-none'
    : 'overflow-y-hidden'}"
  bind:clientWidth={containerWidth}
  bind:clientHeight={containerHeight}
>
  <div
    bind:this={innerContainer}
    style:width="{containerWidth}px"
    style:height="{innerHeight}px"
    style:position="relative"
    style:overflow="hidden"
  >
    {#each newChartRects as rect}
      {@const p = grid.resolvePlacement(rect)}
      <button
        class="absolute rounded-md border bg-slate-100 dark:bg-slate-900 border-slate-400 text-slate-400 dark:border-slate-500 dark:text-slate-500 border-dashed opacity-0 hover:opacity-50 flex items-center justify-center"
        style:left="{p.x}px"
        style:top="{p.y}px"
        style:width="{p.width}px"
        style:height="{p.height}px"
        onclick={newChart.bind(null, rect)}
        transition:fade={{ duration: 100 }}
      >
        <IconPlus />
      </button>
    {/each}

    {#each Object.keys(charts) as id (id)}
      <DashboardChartPanel
        context={context}
        id={id}
        spec={charts[id]}
        placement={placements[id] ?? { x: 0, y: 0, width: 4, height: 3 }}
        grid={grid}
        order={chartIds.indexOf(id)}
        onRemove={removeChart.bind(null, id)}
        onPlacementChange={(placement) => {
          onStateChange({ grids: { [gridKey]: { placements: { ...placements, [id]: placement } } } });
        }}
        onIsPlacementChanging={(value) => (placementChangingCounter += value ? 1 : -1)}
        onBringToFront={bringToFront.bind(null, id)}
        onSpecChange={(spec) => {
          onChartsChange({ [id]: undefined });
          onChartStatesChange({ [id]: undefined });
          onChartsChange({ [id]: spec });
        }}
        chartView={chartView}
      />
    {/each}
  </div>
</div>
