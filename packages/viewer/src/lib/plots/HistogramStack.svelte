<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { Selection, type Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { format } from "d3-format";

  import SizeReader from "../charts/SizeReader.svelte";
  import XYFrame from "../charts/XYFrame.svelte";
  import XYSelection from "../charts/XYSelection.svelte";
  import InlineSelect from "./controls/InlineSelect.svelte";
  import ScaleTypePicker from "./controls/ScaleTypePicker.svelte";

  import { inferColorScale } from "../charts/infer.js";
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
    groupField: string;
    xBinCount?: number;
    groupBinCount?: number;

    filter: Selection;

    stateStore?: PlotStateStore<{
      brush: { x: [number, number] | string } | null;
      xScaleType: ScaleType | null;
      normalization: "x" | null;
    }>;
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

  let { table, xField, groupField, xBinCount = 20, groupBinCount = 20, filter, stateStore }: Props = $props();
  const coordinator = Context.coordinator;
  const darkMode = Context.darkMode;

  let colors = $derived($darkMode ? plotColors.dark : plotColors.light);

  let xScaleType: ScaleType | null = $state.raw(null);
  let normalization: "x" | null = $state.raw(null);
  let brush: { x: AggregateKey } | null = $state.raw(null);

  let chartData = $state.raw<ChartData | null>(null);

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
            { key: "x", stats: stats.x, scaleType: xScaleType, binCount: xBinCount },
            { key: "group", stats: stats.group, binCount: groupBinCount },
          )
        : null,
    );

    $effect.pre(() => {
      if (xScaleType == null) {
        xScaleType = aggregate?.scales.x?.type ?? null;
      }
    });

    function createClient(
      aggregate: DistributionAggregate,
      selection: Selection | null,
      callback: (bins: Record<string, any>[]) => void,
    ) {
      return makeClient({
        coordinator: coordinator,
        selection: selection,
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

      let baseBins: any[] | null = $state.raw(null);
      let selectionBins: any[] | null = $state.raw(null);

      let clientBase = createClient(capturedAggregate, null, (data) => {
        baseBins = data;
      });
      let clientSelection = createClient(capturedAggregate, filter, (data) => {
        selectionBins = data;
      });
      (clientSelection as any).reset = () => {
        brush = null;
      };

      $effect.pre(() => {
        if (baseBins != null && selectionBins != null) {
          let keyfunc = (g: AggregateKey): string => JSON.stringify(g);
          let valueField = normalization == "x" ? "normalizeByX" : "count";

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
          if (normalization) {
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
    initializeClient(coordinator, table, xField, groupField, filter);
  });

  // Sync with state store
  $effect.pre(() =>
    syncState(
      stateStore,
      () => ({
        brush: brush,
        xScaleType: xScaleType,
        normalization: normalization,
      }),
      (value) => {
        brush = value.brush;
        xScaleType = value.xScaleType;
        normalization = value.normalization;
      },
    ),
  );

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

<div style:height="200px">
  <SizeReader>
    {#snippet children(width, height)}
      {#if chartData != null}
        <XYFrame width={width} height={height} xScale={chartData.xScale} yScale={chartData.yScale}>
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
              onChange={(v) => (brush = v != null && v.x != null ? { x: v.x } : null)}
            />
          {/snippet}
        </XYFrame>
      {/if}
    {/snippet}
  </SizeReader>
</div>
<div class="mt-2 flex gap-2 items-start text-sm">
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
    <ScaleTypePicker label="X" bind:value={xScaleType} />
    <span class="flex gap-1 select-none">
      <span class="text-slate-400 dark:text-slate-500 text-sm">Normalize:</span>
      <InlineSelect
        options={[
          { value: null, label: "off" },
          { value: "x", label: "X" },
        ]}
        value={normalization}
        onChange={(v) => (normalization = v)}
      />
    </span>
  </span>
</div>
