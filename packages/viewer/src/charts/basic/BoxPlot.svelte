<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { makeClient, Selection, type Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  import Container from "../common/Container.svelte";
  import SizeReader from "../common/SizeReader.svelte";
  import XYFrame from "../common/XYFrame.svelte";
  import XYSelection from "../common/XYSelection.svelte";
  import ScaleTypePicker from "../controls/ScaleTypePicker.svelte";

  import type { ChartViewProps } from "../chart.js";
  import { chartColors } from "../common/colors.js";
  import {
    distributionAggregate,
    distributionStats,
    type AggregateKey,
    type DistributionAggregate,
    type DistributionStats,
  } from "../common/distribution_helper.js";
  import { inferPositionScale } from "../common/infer.js";
  import type { ScaleSpec, ScaleType } from "../common/types.js";
  import type { BoxPlotSpec } from "./types.js";

  interface State {
    brush?: { x: AggregateKey } | null;
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

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onSpecChange,
    onStateChange,
  }: ChartViewProps<BoxPlotSpec, State> = $props();

  let { colorScheme } = context;

  let { brush } = $derived(chartState);

  let colors = $derived(chartColors[$colorScheme]);

  let chartData: ChartData | null = $state.raw(null);

  let defaultXScaleType = $state.raw<ScaleType | null>(null);
  let defaultYScaleType = $state.raw<ScaleType | null>(null);

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
      stats
        ? distributionAggregate({ key: "x", stats: stats.x, scaleType: spec.xScaleType, binCount: spec.xBinCount })
        : null,
    );

    let yScale = $derived(stats?.y.quantitative ? inferPositionScale(stats.y.quantitative, spec.yScaleType) : null);

    $effect.pre(() => {
      if (defaultXScaleType == null && aggregate?.scales.x?.type != null) {
        defaultXScaleType = aggregate?.scales.x?.type;
      }
      if (defaultYScaleType == null && yScale?.type != null) {
        defaultYScaleType = yScale?.type;
      }
    });

    function createClient(
      aggregate: DistributionAggregate,
      selection: Selection | null,
      callback: (bins: any[]) => void,
    ) {
      return makeClient({
        coordinator: coordinator,
        selection: selection ?? undefined,
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
        queryResult: (data: any) => {
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
        onStateChange({ brush: null });
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
    initializeClient(context.coordinator, context.table, spec.data.x, spec.data.y, context.filter);
  });
</script>

<Container width={width} height={height} class="flex flex-col">
  <div class="text-slate-400 mb-1 select-none">↑ {spec.data.y}</div>
  <SizeReader flexHeight={height != null}>
    {#snippet children(width, height)}
      {#if chartData != null}
        <XYFrame
          width={width}
          height={height}
          xScale={chartData.xScale}
          yScale={chartData.yScale}
          colorScheme={$colorScheme}
        >
          {#snippet children(proxy)}
            {@const xScale = proxy.xScale!}
            {@const yScale = proxy.yScale!}
            {@const lineColor = $colorScheme == "dark" ? "#bbbbbb" : "black"}
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
              value={brush ?? null}
              colorScheme={$colorScheme}
              onChange={(v) => {
                onStateChange({ brush: v != null && v.x != null ? { x: v.x } : null });
              }}
            />
          {/snippet}
        </XYFrame>
      {/if}
    {/snippet}
  </SizeReader>
  <div class="text-slate-400 mb-1 select-none text-right">{spec.data.x} →</div>
  <div class="flex flex-col items-end gap-1">
    <div class="flex flex gap-2 mt-2">
      <ScaleTypePicker
        label="X"
        bind:value={
          () => spec.xScaleType ?? defaultXScaleType ?? "linear",
          (v) => {
            onSpecChange({ xScaleType: v });
          }
        }
      />
      <ScaleTypePicker
        label="Y"
        bind:value={
          () => spec.yScaleType ?? defaultYScaleType ?? "linear",
          (v) => {
            onSpecChange({ yScaleType: v });
          }
        }
      />
    </div>
  </div>
</Container>
