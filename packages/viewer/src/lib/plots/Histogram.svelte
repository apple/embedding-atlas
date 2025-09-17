<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { Selection, type Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { format } from "d3-format";

  import SizeReader from "../charts/SizeReader.svelte";
  import XYFrame from "../charts/XYFrame.svelte";
  import XYSelection from "../charts/XYSelection.svelte";
  import ScaleTypePicker from "./controls/ScaleTypePicker.svelte";

  import type { ScaleSpec, ScaleType } from "../charts/types.js";
  import { Context } from "../contexts.js";
  import { plotColors } from "./colors.js";
  import {
    distributionAggregate,
    distributionStats,
    type AggregateKey,
    type DistributionAggregate,
  } from "./distribution_helper.js";
  import { makeClient } from "./mosaic_helper.js";
  import type { PlotStateStore } from "./plot_state_store.js";
  import { syncState } from "./utils.svelte";

  interface Props {
    table: string;
    field: string;
    binCount?: number;
    filter: Selection;
    stateStore?: PlotStateStore<{
      brush: { x: AggregateKey } | null;
      xScaleType: ScaleType | null;
    }>;
  }

  interface Bin {
    x: AggregateKey;
    count: number;
  }

  interface ChartData {
    xScale: ScaleSpec;
    allItems: Bin[];
    filteredItems: Bin[];
  }

  let { table, field, binCount = 20, filter, stateStore }: Props = $props();
  const coordinator = Context.coordinator;
  const darkMode = Context.darkMode;

  let colors = $derived($darkMode ? plotColors.dark : plotColors.light);

  let xScaleType: ScaleType | null = $state.raw(null);
  let brush: { x: AggregateKey } | null = $state.raw(null);

  let chartData: ChartData | null = $state.raw(null);

  function initializeClient(coordinator: Coordinator, table: string, field: string, filter: Selection) {
    let stats: any | null = $state.raw(null);

    // Query the stats
    distributionStats(coordinator, table, field).then((r) => {
      stats = r;
    });

    // Infer binning from stats
    let aggregate: DistributionAggregate | null = $derived(
      stats ? distributionAggregate({ key: "x", stats: stats, scaleType: xScaleType, binCount: binCount }) : null,
    );

    $effect.pre(() => {
      if (xScaleType == null) {
        xScaleType = aggregate?.scales.x?.type ?? null;
      }
    });

    function createClient(
      aggregate: DistributionAggregate,
      selection: Selection | null,
      callback: (bins: any[]) => void,
    ) {
      return makeClient({
        coordinator: coordinator,
        selection: selection,
        query: (predicate) => {
          return SQL.Query.from(table)
            .select({ ...aggregate.select, count: SQL.count() })
            .where(predicate)
            .groupby(aggregate.select.x);
        },
        queryResult: (data) => {
          callback(Array.from(data).map(aggregate.collect));
        },
      });
    }

    $effect.pre(() => {
      if (aggregate == null) {
        return;
      }
      let capturedAggregate = aggregate;

      let allItems: Bin[] = $state([]);
      let filteredItems: Bin[] = $state([]);

      let clientBase = createClient(capturedAggregate, null, (data) => {
        allItems = data;
      });
      let clientSelection = createClient(capturedAggregate, filter, (data) => {
        filteredItems = data;
      });

      (clientSelection as any).reset = () => {
        brush = null;
      };

      $effect.pre(() => {
        if (allItems.length > 0) {
          chartData = {
            xScale: capturedAggregate.scales.x,
            allItems: allItems,
            filteredItems: filteredItems,
          };
        }
      });

      // Sync selection with brush
      $effect.pre(() => {
        let clause: any = {
          source: clientSelection,
          clients: new Set([clientSelection]),
          ...(brush != null ? capturedAggregate.clause(brush) : { value: null, predicate: null }),
        };
        filter.update(clause);
        filter.activate(clause);
      });

      return () => {
        clientBase.destroy();
        clientSelection.destroy();
        filter.update({
          source: clientSelection,
          clients: new Set([clientSelection]),
          value: null,
          predicate: null,
        });
      };
    });
  }

  $effect.pre(() => {
    initializeClient(coordinator, table, field, filter);
  });

  // Sync with state store
  $effect.pre(() =>
    syncState(
      stateStore,
      () => ({
        brush: brush,
        xScaleType: xScaleType,
      }),
      (value) => {
        brush = value.brush;
        xScaleType = value.xScaleType;
      },
    ),
  );
</script>

<div style:height="200px">
  <SizeReader>
    {#snippet children(width, height)}
      {#if chartData != null}
        {@const maxCount = chartData.allItems.reduce((a, b) => Math.max(a, b.count), 1)}
        <XYFrame
          width={width}
          height={height}
          xScale={chartData.xScale}
          yScale={{ type: "linear", domain: [0, maxCount] }}
        >
          {#snippet children(proxy)}
            {@const xScale = proxy.xScale!}
            {@const yScale = proxy.yScale!}
            {#each chartData?.allItems ?? [] as element}
              {@const [x0, x1] = xScale.applyBand(element.x)}
              {@const [y0, y1] = yScale.applyBand([0, element.count])}
              {@const barGap = Math.min(Math.abs(x1 - x0) * 0.1, 1)}
              <rect
                x={Math.min(x0, x1) + barGap / 2}
                y={Math.min(y0, y1)}
                width={Math.abs(x0 - x1) - barGap}
                height={Math.abs(y0 - y1)}
                fill={colors.markColorFade}
              />
            {/each}
            {#each chartData?.filteredItems ?? [] as element}
              {@const [x0, x1] = xScale.applyBand(element.x)}
              {@const [y0, y1] = yScale.applyBand([0, element.count])}
              {@const barGap = Math.min(Math.abs(x1 - x0) * 0.1, 1)}
              <rect
                x={Math.min(x0, x1) + barGap / 2}
                height={Math.abs(y0 - y1)}
                y={Math.min(y0, y1)}
                width={Math.abs(x0 - x1) - barGap}
                fill={colors.markColor}
              />
            {/each}
            <XYSelection
              proxy={proxy}
              mode="x"
              value={brush}
              onChange={(v) => (brush = v != null && v.x != null ? { x: v.x } : null)}
            />
          {/snippet}
        </XYFrame>
      {/if}
    {/snippet}
  </SizeReader>
</div>
{#if chartData?.xScale.type != "band"}
  <div class="mt-2 flex gap items-center text-sm">
    <span class="flex-1 text-slate-400 dark:text-slate-500">
      {#if brush}
        {#if typeof brush.x == "string"}
          [{brush.x}]
        {:else}
          {@const fmt = format(".4")}
          [{fmt(brush.x[0])}, {fmt(brush.x[1])}]
        {/if}
      {/if}
    </span>
    <ScaleTypePicker label="X" bind:value={xScaleType} />
  </div>
{/if}
