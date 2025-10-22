<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { makeClient, Selection, type Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { interpolateInferno, interpolatePuBuGn } from "d3-scale-chromatic";

  import InlineSelect from "../../widgets/InlineSelect.svelte";
  import Container from "../common/Container.svelte";
  import Raster from "../common/Raster.svelte";
  import SizeReader from "../common/SizeReader.svelte";
  import XYFrame from "../common/XYFrame.svelte";
  import XYSelection from "../common/XYSelection.svelte";
  import ScaleTypePicker from "../controls/ScaleTypePicker.svelte";

  import type { ChartViewProps } from "../chart.js";
  import {
    distributionAggregate,
    distributionStats,
    type AggregateKey,
    type DistributionAggregate,
    type DistributionStats,
  } from "../common/distribution_helper.js";
  import type { ScaleSpec, ScaleType } from "../common/types.js";
  import type { Histogram2DSpec } from "./types.js";

  interface State {
    brush: { x: AggregateKey; y: AggregateKey } | null;
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

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onSpecChange,
    onStateChange,
  }: ChartViewProps<Histogram2DSpec, State> = $props();

  let { colorScheme } = context;

  let { brush } = $derived(chartState);

  let chartData = $state.raw<ChartData | null>(null);
  let maxCount = $derived(chartData?.items.reduce((a, b) => Math.max(a, b.value), 1) ?? 1);
  const zeroShift = 0.07;
  const adjustForZero = (x: number) => (x > 0 ? zeroShift + (1 - zeroShift) * x : 0);
  let interpolateScheme = $derived(
    $colorScheme == "dark"
      ? (x: number) => interpolateInferno(adjustForZero(x / maxCount))
      : (x: number) => interpolatePuBuGn(adjustForZero(x / maxCount)),
  );

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
        ? distributionAggregate(
            { key: "x", stats: stats.x, scaleType: spec.xScaleType, binCount: spec.xBinCount },
            { key: "y", stats: stats.y, scaleType: spec.yScaleType, binCount: spec.yBinCount },
          )
        : null,
    );

    $effect.pre(() => {
      if (defaultXScaleType == null && aggregate?.scales.x?.type != null) {
        defaultXScaleType = aggregate?.scales.x?.type;
      }
      if (defaultYScaleType == null && aggregate?.scales.y?.type != null) {
        defaultYScaleType = aggregate?.scales.y?.type;
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
        queryResult: (data: any) => {
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

      let source = {
        reset: () => {
          onStateChange({ brush: null });
        },
      };

      $effect.pre(() => {
        if (selectionBins != null) {
          chartData = {
            xScale: capturedAggregate.scales.x,
            yScale: capturedAggregate.scales.y,
            items: selectionBins.map((b) => ({
              x: b.x,
              y: b.y,
              value: spec.normalization == "x" ? b.normalizeByX : spec.normalization == "y" ? b.normalizeByY : b.count,
            })),
          };
        }
      });

      // Sync selection with brush
      $effect.pre(() => {
        let clause: any = {
          source: source,
          clients: new Set([clientSelection]),
          ...(brush != null ? capturedAggregate.clause(brush) : { value: null, predicate: null }),
        };
        filter.update(clause);
        filter.activate(clause);
      });

      return () => {
        clientSelection.destroy();
        filter.update({
          source: source,
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
  <SizeReader flexHeight={height != null} defaultHeight={Math.min(300, width ?? 300)}>
    {#snippet children(width, height)}
      {#if chartData != null}
        <XYFrame
          width={width}
          height={height}
          xScale={chartData.xScale}
          yScale={chartData.yScale}
          colorScheme={$colorScheme}
        >
          {#snippet childrenBelow(proxy)}
            <rect x={0} y={0} width={proxy.plotWidth} height={proxy.plotHeight} fill={interpolateScheme(0)} />
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
                fill={interpolateScheme(item.value)}
              />
            {/each}
          {/snippet}
          {#snippet children(proxy)}
            <XYSelection
              proxy={proxy}
              mode="xy"
              value={brush}
              colorScheme={$colorScheme}
              onChange={(v) => {
                onStateChange({
                  brush: v != null && v.x != null && v.y != null ? { x: v.x, y: v.y } : null,
                });
              }}
            />
          {/snippet}
        </XYFrame>
      {/if}
    {/snippet}
  </SizeReader>
  <div class="text-slate-400 mb-1 select-none text-right">{spec.data.x} →</div>
  <div class="flex gap items-center text-sm">
    <span class="flex-1 text-slate-400 dark:text-slate-500">
      <XYFrame
        xScale={{ type: "linear", domain: [0, maxCount] }}
        xAxis={{ extendScaleToTicks: false }}
        width={230}
        height={24}
        extents={{ left: 30, right: 30, top: 0, bottom: 0 }}
        colorScheme={$colorScheme}
      >
        {#snippet children(proxy)}
          <Raster
            color={interpolateScheme}
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
      </span>
      <span class="flex gap-1 select-none">
        <span class="text-slate-400 dark:text-slate-500 text-sm">Normalize:</span>
        <InlineSelect
          options={[
            { value: null, label: "off" },
            { value: "x", label: "X" },
            { value: "y", label: "Y" },
          ]}
          value={spec.normalization ?? null}
          onChange={(v) => onSpecChange({ normalization: v })}
        />
      </span>
    </span>
  </div>
</Container>
