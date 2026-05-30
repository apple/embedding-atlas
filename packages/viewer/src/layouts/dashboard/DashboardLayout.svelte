<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { fade } from "svelte/transition";

  import DashboardChartPanel from "./DashboardChartPanel.svelte";

  import { IconPlus } from "../../assets/icons.js";

  import { deepMemo } from "../../../../utils/dist/equals.js";
  import { getStoreContext } from "../../stores/embedding_atlas_store.js";
  import { reorder } from "../../utils/sort.js";
  import type { LayoutProps } from "../layout.js";
  import { OccupancyMap } from "./occupancy_map.js";
  import { Grid, computePlacements, overlaps } from "./placement.js";
  import type { DashboardLayoutSpec, Placement } from "./types.js";

  const { layout, chartView }: LayoutProps = $props();

  const store = getStoreContext();
  const { layouts, charts } = store;
  let spec = $derived($layouts[layout] ?? {}) as DashboardLayoutSpec;

  let containerWidth = $state(100);
  let containerHeight = $state(100);

  let innerContainer: HTMLDivElement;

  let placementChangingCounter = $state(0);

  let visibleCharts = $derived.by(() => {
    const out: Record<string, any> = {};
    for (const id of spec.chartIds ?? []) {
      if ($charts[id] != null) out[id] = $charts[id];
    }
    return out;
  });

  let numColumns = $derived(spec.numColumns ?? (containerWidth < 500 ? 8 : 24));
  let numRows = $derived(spec.numRows ?? 18);
  let grid = $derived(new Grid(containerWidth, containerHeight, numColumns, numRows, 6));
  let gridKey = $derived(`${numColumns}x${numRows}`);

  let placements = $derived.by(() =>
    computePlacements(visibleCharts, spec.grids?.[gridKey]?.placements ?? {}, grid.numColumns),
  );

  let orderingInfo = $derived.by(
    deepMemo(() => {
      let order = reorder(Object.keys(visibleCharts), spec.grids?.[gridKey]?.order).reverse();

      return Object.fromEntries(
        order.map((id, index) => {
          let hasOverlap = false;
          let p1 = placements[id];
          for (let rid of order.slice(0, index)) {
            if (overlaps(p1, placements[rid])) {
              hasOverlap = true;
              break;
            }
          }
          return [id, { order: index, hasOverlap: hasOverlap }];
        }),
      );
    }),
  );

  let newChartRects = $derived.by(() => {
    let map = new OccupancyMap(grid.numColumns);
    for (let id in placements) {
      map.fill(placements[id].x, placements[id].y, placements[id].width, placements[id].height);
    }
    return map.unusedRects(4, 4, 8, 6, numRows);
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
  let innerHeight = $derived(grid.totalHeight(effectiveMaxY));

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
    store.removeChartFromLayout(layout, id);
  }

  function bringToFront(id: string) {
    let existingOrder = spec.grids?.[gridKey]?.order ?? [];
    let newOrder = [id, ...existingOrder.filter((x: string) => x != id)];
    store.updateLayout<DashboardLayoutSpec>(layout, (draft) => {
      draft.grids ??= {};
      draft.grids[gridKey] ??= {};
      draft.grids[gridKey].order = newOrder;
    });
  }

  function newChart(placement: Placement) {
    let id = store.addChartToLayout(layout);
    store.updateLayout<DashboardLayoutSpec>(layout, (draft) => {
      draft.grids ??= {};
      draft.grids[gridKey] ??= {};
      draft.grids[gridKey].placements ??= {};
      draft.grids[gridKey].placements[id] = placement;
    });
  }

  let chartPanels = $state<Record<string, DashboardChartPanel | null>>({});
  onMount(() => {
    let chartIDs = new Set(Object.keys(visibleCharts));
    $effect(() => {
      let oldIDs = chartIDs;
      chartIDs = new Set(Object.keys(visibleCharts));
      if (chartIDs.size != oldIDs.size + 1) {
        return;
      }
      let diff: string[] = [];
      for (let id of chartIDs) {
        if (!oldIDs.has(id)) {
          diff.push(id);
        }
      }
      if (diff.length == 1) {
        chartPanels[diff[0]]?.scrollIntoView();
      }
    });
  });
</script>

<div
  class="w-full h-full overflow-x-hidden relative {innerHeight > containerHeight
    ? 'overflow-y-scroll overscroll-none'
    : 'overflow-y-hidden'}"
  bind:clientHeight={containerHeight}
>
  <div
    bind:this={innerContainer}
    style:z-index={0}
    bind:clientWidth={containerWidth}
    style:width="100%"
    style:height="{innerHeight}px"
    style:position="relative"
  >
    {#each newChartRects as rect}
      {@const p = grid.resolvePlacement(rect)}
      <button
        class="absolute rounded-md border bg-slate-100 dark:bg-slate-900 border-slate-400 text-slate-600 dark:border-slate-500 dark:text-slate-400 border-dashed opacity-0 hover:opacity-50 flex items-center justify-center"
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

    {#each Object.keys(visibleCharts) as id (id)}
      <DashboardChartPanel
        bind:this={chartPanels[id]}
        context={store.chartContext}
        id={id}
        spec={$charts[id]}
        placement={placements[id] ?? { x: 0, y: 0, width: 4, height: 3 }}
        grid={grid}
        order={orderingInfo[id].order}
        hasBorder={orderingInfo[id].hasOverlap}
        onRemove={removeChart.bind(null, id)}
        onPlacementChange={(placement) => {
          store.updateLayout<DashboardLayoutSpec>(layout, (draft) => {
            draft.grids ??= {};
            draft.grids[gridKey] ??= {};
            draft.grids[gridKey].placements = {
              ...placements,
              [id]: placement,
            };
          });
        }}
        onIsPlacementChanging={(value) => (placementChangingCounter += value ? 1 : -1)}
        onBringToFront={bringToFront.bind(null, id)}
        onSpecChange={(spec) => {
          store.replaceChart(id, spec);
        }}
        chartView={chartView}
      />
    {/each}
  </div>
</div>
