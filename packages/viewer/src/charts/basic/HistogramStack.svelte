<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { makeClient, Selection, type Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { format } from "d3-format";

  import InlineSelect from "../../widgets/InlineSelect.svelte";
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
  import { inferColorScale } from "../common/infer.js";
  import type { ScaleSpec, ScaleType } from "../common/types.js";
  import type { HistogramStackSpec } from "./types.js";

  interface State {
    brush: { x: [number, number] | string } | null;
  }

  interface ChartData {
    xScale: ScaleSpec;
    yScale: ScaleSpec;
    colorScale: {
      domain: AggregateKey[];
      apply: (value: AggregateKey) => string;
    };
    totals: { x: AggregateKey; total: number }[];
    items: { x: AggregateKey; groups: { group: AggregateKey; y1: number; y2: number }[] }[];
  }

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onSpecChange,
    onStateChange,
  }: ChartViewProps<HistogramStackSpec, State> = $props();

  let { colorScheme } = context;

  let { brush } = $derived(chartState);

  let colors = $derived(chartColors[$colorScheme]);

  let chartData = $state.raw<ChartData | null>(null);
  let defaultScaleType = $state<ScaleType | null>(null);

  function initializeClient(
    coordinator: Coordinator,
    table: string,
    xField: string,
    groupField: string,
    filter: Selection,
  ) {
    let stats = $state.raw<{ x: DistributionStats; group: DistributionStats } | null>(null);

    // Query the stats
    Promise.all([
      distributionStats(coordinator, table, xField),
      distributionStats(coordinator, table, groupField),
    ]).then(([x, group]) => {
      stats = x != null && group != null ? { x, group } : null;
    });

    // Infer binning from stats
    let aggregate: DistributionAggregate | null = $derived(
      stats
        ? distributionAggregate(
            { key: "x", stats: stats.x, scaleType: spec.xScaleType, binCount: spec.xBinCount },
            { key: "group", stats: stats.group, binCount: spec.groupBinCount },
          )
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
      callback: (bins: Record<string, any>[]) => void,
    ) {
      return makeClient({
        coordinator: coordinator,
        selection: selection ?? undefined,
        query: (predicate) => {
          return SQL.Query.from(
            SQL.Query.from(table)
              .select({ ...aggregate.select, count: SQL.count() })
              .where(predicate)
              .groupby(aggregate.select.x, aggregate.select.group),
          ).select({
            x: "x",
            group: "group",
            count: "count",
            normalizeByX: SQL.sql`count / (SUM(count) OVER (PARTITION BY x))`,
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

      let baseBins: any[] | null = $state.raw(null);
      let selectionBins: any[] | null = $state.raw(null);

      let clientBase = createClient(capturedAggregate, null, (data) => {
        baseBins = data;
      });
      let clientSelection = createClient(capturedAggregate, filter, (data) => {
        selectionBins = data;
      });
      let source = {
        reset: () => {
          onStateChange({ brush: null });
        },
      };

      $effect.pre(() => {
        if (baseBins != null && selectionBins != null) {
          let keyfunc = (g: AggregateKey): string => JSON.stringify(g);
          let valueField = spec.normalization == "x" ? "normalizeByX" : "count";

          let groupLevels = Array.from(
            collect(
              baseBins,
              (v) => keyfunc(v.group),
              (items) => items[0].group,
            ).entries(),
          ).sort((a, b) => capturedAggregate.order.group(a[1], b[1]));

          let totals = collect(
            baseBins,
            (v) => keyfunc(v.x),
            (items) => ({ x: items[0].x, total: items.reduce((a, b) => a + b[valueField], 0) }),
          );
          let items = collect(
            selectionBins,
            (v) => keyfunc(v.x),
            (items) => {
              items = items.sort((a, b) => capturedAggregate.order.group(a.group, b.group));
              let result = [];
              let csum = 0;
              for (let item of items) {
                result.push({
                  group: item.group,
                  y1: csum,
                  y2: csum + item[valueField],
                });
                csum += item[valueField];
              }
              return { x: items[0].x, groups: result };
            },
          );
          let maxCount = totals.values().reduce((a, b) => Math.max(a, b.total), 1);
          if (spec.normalization) {
            maxCount = 1;
          }

          let colorScale = inferColorScale(Array.from(groupLevels.map((x) => x[1])), {
            fade: ["n/a", "(null)"],
            ordinal: capturedAggregate.scales.group.type != "band",
          });

          chartData = {
            xScale: capturedAggregate.scales.x,
            yScale: { type: "linear", domain: [0, maxCount] },
            colorScale: colorScale,
            totals: Array.from(totals.values()),
            items: Array.from(items.values()),
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
    initializeClient(context.coordinator, context.table, spec.data.x, spec.data.group, context.filter);
  });

  function collect<T, K, U>(inputs: T[], key: (item: T) => K, mapItems: (items: T[]) => U): Map<K, U> {
    let result = new Map<K, T[]>();
    for (let item of inputs) {
      let k = key(item);
      let list = result.get(k);
      if (!list) {
        list = [];
        result.set(k, list);
      }
      list.push(item);
    }
    return new Map(result.entries().map(([k, v]) => [k, mapItems(v)]));
  }

  function group2string(group: AggregateKey): string {
    if (typeof group == "string") {
      return group;
    } else {
      let fmt = format(".6");
      if (group.length == 2) {
        return `[${fmt(group[0])}, ${fmt(group[1])})`;
      }
    }
    return "(invalid)";
  }
</script>

<Container width={width} height={height} class="flex flex-col gap-2">
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
            {#each chartData?.totals ?? [] as item}
              {@const [x0, x1] = xScale.applyBand(item.x)}
              {@const [y0, y1] = yScale.applyBand([0, item.total])}
              {@const gap = Math.min(Math.abs(x1 - x0) * 0.2, Math.abs(y1 - y0) * 0.2, 1)}
              <rect
                x={Math.min(x0, x1) + gap / 2}
                y={Math.min(y0, y1)}
                width={Math.abs(x0 - x1) - gap}
                height={Math.abs(y0 - y1)}
                fill={colors.markColorFade}
              />
            {/each}
            {#each chartData?.items ?? [] as { x, groups }}
              {@const [x0, x1] = xScale.applyBand(x)}
              {#each groups as item}
                {@const [y0, y1] = yScale.applyBand([item.y1, item.y2])}
                {@const gap = Math.min(Math.abs(x1 - x0) * 0.2, Math.abs(y1 - y0) * 0.2, 1)}
                <rect
                  x={Math.min(x0, x1) + gap / 2}
                  y={Math.min(y0, y1)}
                  width={Math.abs(x0 - x1) - gap}
                  height={Math.abs(y0 - y1)}
                  fill={chartData?.colorScale.apply(item.group)}
                />
              {/each}
            {/each}
            <XYSelection
              proxy={proxy}
              mode="x"
              value={brush}
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
  <div class="flex-none flex gap-2 items-start text-sm">
    <div class="flex-1 text-slate-400 dark:text-slate-500">
      {#if chartData}
        <div class="flex gap-2 flex-wrap items-center select-none">
          {#each chartData.colorScale.domain as group}
            <div class="flex gap-1 items-center" title={JSON.stringify(group)}>
              <div class="w-3 h-3 block rounded-sm" style:background={chartData.colorScale.apply(group)}></div>
              <div class="whitespace-nowrap max-w-32 overflow-hidden text-ellipsis">
                {group2string(group)}
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
    <span class="flex flex-col items-end gap-1">
      <ScaleTypePicker
        label="X"
        bind:value={
          () => spec.xScaleType ?? defaultScaleType ?? "linear",
          (v) => {
            onSpecChange({ xScaleType: v });
          }
        }
      />
      <span class="flex gap-1 select-none">
        <span class="text-slate-400 dark:text-slate-500 text-sm">Normalize:</span>
        <InlineSelect
          options={[
            { value: null, label: "off" },
            { value: "x", label: "X" },
          ]}
          value={spec.normalization}
          onChange={(v) => onSpecChange({ normalization: v })}
        />
      </span>
    </span>
  </div>
</Container>
