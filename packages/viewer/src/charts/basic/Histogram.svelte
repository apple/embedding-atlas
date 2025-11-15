<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { makeClient, Selection, type Coordinator, type SelectionClause } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { format } from "d3-format";

  import Container from "../common/Container.svelte";
  import SizeReader from "../common/SizeReader.svelte";
  import XYFrame from "../common/XYFrame.svelte";
  import XYSelection from "../common/XYSelection.svelte";
  import ScaleTypePicker from "../controls/ScaleTypePicker.svelte";

  import type { ChartViewProps } from "../chart.js";
  import { getChartColors } from "../common/colors.js";
  import {
    distributionAggregate,
    distributionStats,
    type AggregateKey,
    type DistributionAggregate,
    type DistributionStats,
  } from "../common/distribution_helper.js";
  import type { ScaleSpec, ScaleType } from "../common/types.js";
  import type { HistogramSpec } from "./types.js";

  interface State {
    brush?: { x: AggregateKey } | null;
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

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onSpecChange,
    onStateChange,
  }: ChartViewProps<HistogramSpec, State> = $props();

  let { coordinator, colorScheme, customChartColors } = context;

  let { brush } = $derived(chartState);

  let colors = $derived(getChartColors($colorScheme, customChartColors));

  let chartData: ChartData | null = $state.raw(null);
  let defaultScaleType = $state<ScaleType | null>(null);

  function initializeClient(coordinator: Coordinator, table: string, field: string, filter: Selection) {
    let stats: DistributionStats | null = $state.raw(null);

    // Query the stats
    distributionStats(coordinator, table, field).then((r) => {
      stats = r;
    });

    // Infer binning from stats
    let aggregate: DistributionAggregate | null = $derived(
      stats
        ? distributionAggregate({ key: "x", stats: stats, scaleType: spec.xScaleType, binCount: spec.binCount })
        : null,
    );

    $effect.pre(() => {
      if (defaultScaleType == null && aggregate?.scales.x?.type != null) {
        defaultScaleType = aggregate?.scales.x?.type;
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
          return SQL.Query.from(table)
            .select({ ...aggregate.select, count: SQL.count() })
            .where(predicate)
            .groupby(aggregate.select.x);
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

      let allItems: Bin[] = $state([]);
      let filteredItems: Bin[] = $state([]);

      let clientBase = createClient(capturedAggregate, null, (data) => {
        allItems = data;
      });
      let clientSelection = createClient(capturedAggregate, filter, (data) => {
        filteredItems = data;
      });

      $effect.pre(() => {
        if (allItems.length > 0) {
          chartData = {
            xScale: capturedAggregate.scales.x,
            allItems: allItems,
            filteredItems: filteredItems,
          };
        }
      });

      let source = {
        reset: () => {
          onStateChange({ brush: null });
        },
      };

      // Sync selection with brush
      $effect.pre(() => {
        let clause: SelectionClause = {
          source: source,
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
          source: source,
          clients: new Set([clientSelection]),
          value: null,
          predicate: null,
        });
      };
    });
  }

  $effect.pre(() => {
    initializeClient(coordinator, context.table, spec.data.field, context.filter);
  });
</script>

<Container width={width} height={height} class="flex flex-col gap-2">
  <SizeReader flexHeight={height != null}>
    {#snippet children(width, height)}
      {#if chartData != null}
        {@const maxCount = chartData.allItems.reduce((a, b) => Math.max(a, b.count), 1)}
        <XYFrame
          width={width}
          height={height}
          xScale={chartData.xScale}
          yScale={{ type: "linear", domain: [0, maxCount] }}
          colorScheme={$colorScheme}
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
  {#if chartData?.xScale.type != "band"}
    <div class="flex gap items-center text-sm">
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
      <ScaleTypePicker
        label="X"
        bind:value={
          () => spec.xScaleType ?? defaultScaleType ?? "linear",
          (v) => {
            onSpecChange({ xScaleType: v });
          }
        }
      />
    </div>
  {/if}
</Container>
