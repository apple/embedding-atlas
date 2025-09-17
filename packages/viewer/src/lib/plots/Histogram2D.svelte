<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { Selection, type Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { interpolateInferno, interpolatePuBuGn } from "d3-scale-chromatic";

  import Raster from "../charts/Raster.svelte";
  import SizeReader from "../charts/SizeReader.svelte";
  import XYFrame from "../charts/XYFrame.svelte";
  import XYSelection from "../charts/XYSelection.svelte";
  import InlineSelect from "./controls/InlineSelect.svelte";
  import ScaleTypePicker from "./controls/ScaleTypePicker.svelte";

  import type { ScaleSpec, ScaleType } from "../charts/types.js";
  import { Context } from "../contexts.js";
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
    yField: string;
    xBinCount?: number;
    yBinCount?: number;

    filter: Selection;
    stateStore?: PlotStateStore<{
      brush: { x: AggregateKey; y: AggregateKey } | null;
      xScaleType: ScaleType | null;
      yScaleType: ScaleType | null;
      normalization: "x" | "y" | null;
    }>;
  }

  interface Bin {
    x: AggregateKey;
    y: AggregateKey;
    value: number;
  }

  interface ChartData {
    xScale: ScaleSpec;
    yScale: ScaleSpec;
    items: Bin[];
  }

  let { table, xField, yField, xBinCount = 20, yBinCount = 20, filter, stateStore }: Props = $props();
  const coordinator = Context.coordinator;
  const darkMode = Context.darkMode;

  let xScaleType: ScaleType | null = $state.raw(null);
  let yScaleType: ScaleType | null = $state.raw(null);
  let normalization: "x" | "y" | null = $state.raw(null);
  let brush: { x: AggregateKey; y: AggregateKey } | null = $state.raw(null);

  let chartData = $state.raw<ChartData | null>(null);
  let maxCount = $derived(chartData?.items.reduce((a, b) => Math.max(a, b.value), 1) ?? 1);
  const zeroShift = 0.07;
  const adjustForZero = (x: number) => (x > 0 ? zeroShift + (1 - zeroShift) * x : 0);
  let colorScheme = $derived(
    $darkMode
      ? (x: number) => interpolateInferno(adjustForZero(x / maxCount))
      : (x: number) => interpolatePuBuGn(adjustForZero(x / maxCount)),
  );

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
        ? distributionAggregate(
            { key: "x", stats: stats.x, scaleType: xScaleType, binCount: xBinCount },
            { key: "y", stats: stats.y, scaleType: yScaleType, binCount: yBinCount },
          )
        : null,
    );

    $effect.pre(() => {
      if (xScaleType == null) {
        xScaleType = aggregate?.scales.x?.type ?? null;
      }
      if (yScaleType == null) {
        yScaleType = aggregate?.scales.y?.type ?? null;
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
          return SQL.Query.from(
            SQL.Query.from(table)
              .select({ ...aggregate.select, count: SQL.count() })
              .where(predicate)
              .groupby(aggregate.select.x, aggregate.select.y),
          ).select({
            x: "x",
            y: "y",
            count: "count",
            normalizeByX: SQL.sql`count / (SUM(count) OVER (PARTITION BY x))`,
            normalizeByY: SQL.sql`count / (SUM(count) OVER (PARTITION BY y))`,
          });
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

      let selectionBins: any[] | null = $state.raw(null);

      let clientSelection = createClient(capturedAggregate, filter, (data) => {
        selectionBins = data;
      });
      (clientSelection as any).reset = () => {
        brush = null;
      };

      $effect.pre(() => {
        if (selectionBins != null) {
          chartData = {
            xScale: capturedAggregate.scales.x,
            yScale: capturedAggregate.scales.y,
            items: selectionBins.map((b) => ({
              x: b.x,
              y: b.y,
              value: normalization == "x" ? b.normalizeByX : normalization == "y" ? b.normalizeByY : b.count,
            })),
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
        normalization: normalization,
      }),
      (value) => {
        brush = value.brush;
        xScaleType = value.xScaleType;
        yScaleType = value.yScaleType;
        normalization = value.normalization;
      },
    ),
  );
</script>

<div class="text-slate-400 mb-1 select-none">↑ {yField}</div>
<div style:height="320px">
  <SizeReader>
    {#snippet children(width, height)}
      {#if chartData != null}
        <XYFrame width={width} height={height} xScale={chartData.xScale} yScale={chartData.yScale}>
          {#snippet childrenBelow(proxy)}
            <rect x={0} y={0} width={proxy.plotWidth} height={proxy.plotHeight} fill={colorScheme(0)} />
            {@const xScale = proxy.xScale!}
            {@const yScale = proxy.yScale!}
            {#each chartData?.items ?? [] as item}
              {@const [x0, x1] = xScale.applyBand(item.x)}
              {@const [y0, y1] = yScale.applyBand(item.y)}
              {@const gap = 0}
              <rect
                x={Math.min(x0, x1) + gap / 2}
                y={Math.min(y0, y1) + gap / 2}
                width={Math.abs(x0 - x1) - gap}
                height={Math.abs(y0 - y1) - gap}
                fill={colorScheme(item.value)}
              />
            {/each}
          {/snippet}
          {#snippet children(proxy)}
            <XYSelection
              proxy={proxy}
              mode="xy"
              value={brush}
              onChange={(v) => (brush = v != null && v.x != null && v.y != null ? { x: v.x, y: v.y } : null)}
            />
          {/snippet}
        </XYFrame>
      {/if}
    {/snippet}
  </SizeReader>
</div>
<div class="text-slate-400 mb-1 select-none text-right">{xField} →</div>
<div class="flex gap items-center text-sm">
  <span class="flex-1 text-slate-400 dark:text-slate-500">
    <XYFrame
      xScale={{ type: "linear", domain: [0, maxCount] }}
      xAxis={{ extendScaleToTicks: false }}
      width={230}
      height={24}
      extents={{ left: 30, right: 30, top: 0, bottom: 0 }}
    >
      {#snippet children(proxy)}
        <Raster
          color={colorScheme}
          rasterWidth={100}
          rasterHeight={1}
          proxy={proxy}
          xDomain={proxy.xScale?.domain as any}
        />
      {/snippet}
    </XYFrame>
  </span>
  <span class="flex flex-col items-end gap-1">
    <span class="flex gap-2">
      <ScaleTypePicker label="X" bind:value={xScaleType} />
      <ScaleTypePicker label="Y" bind:value={yScaleType} />
    </span>
    <span class="flex gap-1 select-none">
      <span class="text-slate-400 dark:text-slate-500 text-sm">Normalize:</span>
      <InlineSelect
        options={[
          { value: null, label: "off" },
          { value: "x", label: "X" },
          { value: "y", label: "Y" },
        ]}
        value={normalization}
        onChange={(v) => (normalization = v)}
      />
    </span>
  </span>
</div>
