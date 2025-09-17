<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { Selection, type Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  import SizeReader from "../charts/SizeReader.svelte";
  import XYFrame from "../charts/XYFrame.svelte";
  import XYSelection from "../charts/XYSelection.svelte";
  import ScaleTypePicker from "./controls/ScaleTypePicker.svelte";

  import { inferPositionScale } from "../charts/infer.js";
  import type { ScaleSpec, ScaleType } from "../charts/types.js";
  import { Context } from "../contexts.js";
  import { plotColors } from "./colors.js";
  import {
    distributionAggregate,
    distributionStats,
    type AggregateKey,
    type DistributionAggregate,
    type DistributionStats,
  } from "./distribution_helper.js";
  import { makeClient } from "./mosaic_helper.js";
  import type { PlotStateStore } from "./plot_state_store.js";
  import { syncState } from "./utils.svelte";

  interface Props {
    table: string;
    xField: string;
    xBinCount?: number;
    yField: string;
    filter: Selection;
    stateStore?: PlotStateStore<{
      brush: { x: AggregateKey } | null;
      xScaleType: ScaleType | null;
      yScaleType: ScaleType | null;
    }>;
  }

  interface Bin {
    x: AggregateKey;
    min: number;
    max: number;
    p25: number;
    p50: number;
    p75: number;
  }

  interface ChartData {
    xScale: ScaleSpec;
    yScale: ScaleSpec;
    items: Bin[];
  }

  let { table, xField, xBinCount = 20, yField, filter, stateStore }: Props = $props();
  const coordinator = Context.coordinator;
  const darkMode = Context.darkMode;

  let colors = $derived($darkMode ? plotColors.dark : plotColors.light);

  let xScaleType: ScaleType | null = $state.raw(null);
  let yScaleType: ScaleType | null = $state.raw(null);
  let brush: { x: AggregateKey } | null = $state.raw(null);

  let chartData: ChartData | null = $state.raw(null);

  function initializeClient(
    coordinator: Coordinator,
    table: string,
    xField: string,
    yField: string,
    filter: Selection,
  ) {
    let stats = $state.raw<{ x: DistributionStats; y: DistributionStats } | null>(null);

    // Query the stats
    Promise.all([distributionStats(coordinator, table, xField), distributionStats(coordinator, table, yField)]).then(
      ([x, y]) => {
        stats = x != null && y != null ? { x, y } : null;
      },
    );

    // Infer binning from stats
    let aggregate: DistributionAggregate | null = $derived(
      stats ? distributionAggregate({ key: "x", stats: stats.x, scaleType: xScaleType, binCount: xBinCount }) : null,
    );

    let yScale = $derived(stats?.y.quantitative ? inferPositionScale(stats.y.quantitative, yScaleType) : null);

    $effect.pre(() => {
      if (xScaleType == null) {
        xScaleType = aggregate?.scales.x?.type ?? null;
      }
      if (yScaleType == null) {
        yScaleType = yScale?.type ?? null;
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
          let yExpr = SQL.column(yField, table);
          return SQL.Query.from(table)
            .select({
              ...aggregate.select,
              min: SQL.min(yExpr),
              max: SQL.max(yExpr),
              p50: SQL.median(yExpr),
              p25: SQL.quantile(yExpr, 0.25),
              p75: SQL.quantile(yExpr, 0.75),
            })
            .where(predicate, SQL.isFinite(yExpr))
            .groupby(aggregate.select.x);
        },
        queryResult: (data) => {
          callback(Array.from(data).map(aggregate.collect));
        },
      });
    }

    $effect.pre(() => {
      if (aggregate == null || yScale == null) {
        return;
      }
      let capturedAggregate = aggregate;

      let filteredItems: Bin[] = $state([]);

      let clientSelection = createClient(capturedAggregate, filter, (data) => {
        filteredItems = data;
      });
      (clientSelection as any).reset = () => {
        brush = null;
      };

      $effect.pre(() => {
        chartData = {
          xScale: capturedAggregate.scales.x,
          yScale: yScale,
          items: filteredItems,
        };
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
    initializeClient(coordinator, table, xField, yField, filter);
  });

  // Sync with state store
  $effect.pre(() =>
    syncState(
      stateStore,
      () => ({
        brush: brush,
        xScaleType: xScaleType,
        yScaleType: yScaleType,
      }),
      (value) => {
        brush = value.brush;
        xScaleType = value.xScaleType;
        yScaleType = value.yScaleType;
      },
    ),
  );
</script>

<div class="text-slate-400 mb-1 select-none">↑ {yField}</div>
<div style:height="250px">
  <SizeReader>
    {#snippet children(width, height)}
      {#if chartData != null}
        <XYFrame width={width} height={height} xScale={chartData.xScale} yScale={chartData.yScale}>
          {#snippet children(proxy)}
            {@const xScale = proxy.xScale!}
            {@const yScale = proxy.yScale!}
            {@const lineColor = $darkMode ? "#bbbbbb" : "black"}
            {#each chartData?.items ?? [] as item}
              {@const [x0, x1] = xScale.applyBand(item.x)}
              {@const ym = yScale.apply(item.p50)}
              {@const [ey0, ey1] = yScale.applyBand([item.min, item.max])}
              {@const [by0, by1] = yScale.applyBand([item.p25, item.p75])}
              {@const barGap = Math.min(Math.abs(x1 - x0) * 0.1, 1)}
              {@const lw = Math.abs(x1 - x0) / 3}
              <line y1={ey0} y2={ey1} x1={(x0 + x1) / 2} x2={(x0 + x1) / 2} stroke={lineColor} />
              <line y1={ey0} y2={ey0} x1={(x0 + x1) / 2 - lw / 2} x2={(x0 + x1) / 2 + lw / 2} stroke={lineColor} />
              <line y1={ey1} y2={ey1} x1={(x0 + x1) / 2 - lw / 2} x2={(x0 + x1) / 2 + lw / 2} stroke={lineColor} />
              <rect
                x={Math.min(x0, x1) + barGap / 2}
                height={Math.abs(by0 - by1)}
                y={Math.min(by0, by1)}
                width={Math.abs(x0 - x1) - barGap}
                fill={colors.markColor}
              />
              <line
                y1={ym}
                y2={ym}
                x1={Math.min(x0, x1) + barGap / 2}
                x2={Math.max(x0, x1) - barGap / 2}
                stroke={lineColor}
                stroke-linecap="butt"
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
<div class="text-slate-400 mb-1 select-none text-right">{xField} →</div>

<div class="flex flex-col items-end gap-1">
  <div class="flex flex gap-2 mt-2">
    <ScaleTypePicker label="X" bind:value={xScaleType} />
    <ScaleTypePicker label="Y" bind:value={yScaleType} />
  </div>
</div>
