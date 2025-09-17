<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Coordinator, Selection } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { format } from "d3-format";
  import { scaleLinear } from "d3-scale";

  import InlineSelect from "./controls/InlineSelect.svelte";

  import { Context } from "../contexts.js";
  import { plotColors } from "./colors.js";
  import { distributionAggregate, distributionStats, type DistributionAggregate } from "./distribution_helper.js";
  import { makeClient } from "./mosaic_helper.js";
  import type { PlotStateStore } from "./plot_state_store.js";
  import { syncState } from "./utils.svelte";

  const MAX_BARS = 10;
  const MAX_BARS_EXPANDED = 100;

  interface Props {
    table: string;
    field: string;
    filter: Selection;
    stateStore?: PlotStateStore<{ selection: string[] | null; expanded: boolean; percentage: boolean }>;
  }

  let { table, field, filter, stateStore }: Props = $props();
  const coordinator = Context.coordinator;
  const darkMode = Context.darkMode;

  interface Bin {
    x: string;
    count: number;
  }

  interface ChartData {
    items: { x: string; total: number; selected: number }[];
    sumTotal: number;
    sumSelected: number;
    firstSpecialIndex: number;
    hasOther: boolean;
  }

  let selection: string[] | null = $state.raw(null);
  let expanded = $state.raw(false);
  let percentage = $state.raw(false);

  let chartData = $state.raw<ChartData | null>(null);
  let chartWidth = $state.raw(400);

  let maxCount = $derived(chartData?.items.reduce((a, b) => Math.max(a, percentage ? b.selected : b.total), 0) ?? 0);
  let xScale = $derived(scaleLinear([0, Math.max(1, maxCount)], [0, chartWidth - 250]));

  // Adjust scale so the minimum width for non-zero count is 1px.
  let xScaleAdjusted = $derived((v: number) => (v != 0 ? Math.max(1, xScale(v)) : 0));

  let colors = $derived($darkMode ? plotColors.dark : plotColors.light);

  function initializeClient(coordinator: Coordinator, table: string, field: string, filter: Selection) {
    let stats: any | null = $state.raw(null);

    // Query the stats
    distributionStats(coordinator, table, field).then((r) => {
      stats = r;
    });

    // Infer binning from stats
    let aggregate: DistributionAggregate | null = $derived(
      stats
        ? distributionAggregate({ key: "x", stats: stats, binCount: expanded ? MAX_BARS_EXPANDED : MAX_BARS })
        : null,
    );

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

      let allItems: Bin[] = $state.raw([]);
      let filteredItems: Bin[] = $state.raw([]);

      let clientBase = createClient(capturedAggregate, null, (data) => {
        allItems = data;
      });
      let clientSelection = createClient(capturedAggregate, filter, (data) => {
        filteredItems = data;
      });
      (clientSelection as any).reset = () => {
        selection = null;
      };

      $effect.pre(() => {
        if (allItems.length > 0) {
          let keyfunc = (x: any) => JSON.stringify(x);
          let mapTotal = new Map<string, number>(allItems.map(({ x, count }) => [keyfunc(x), count]));
          let mapSelected = new Map<string, number>(filteredItems.map(({ x, count }) => [keyfunc(x), count]));

          if (allItems.every((d) => typeof d.x == "string")) {
            let specialValues = capturedAggregate.scales.x.specialValues ?? [];
            let hasOther = specialValues.filter((x) => x != "(null)").length > 0;
            let items = [...capturedAggregate.scales.x.domain, ...specialValues].map((d) => ({
              x: d,
              total: mapTotal.get(keyfunc(d)) ?? 0,
              selected: mapSelected.get(keyfunc(d)) ?? 0,
            }));
            let sumTotal = items.reduce((a, b) => a + b.total, 0);
            let sumSelected = items.reduce((a, b) => a + b.selected, 0);
            chartData = {
              items: items,
              sumTotal: sumTotal,
              sumSelected: sumSelected,
              firstSpecialIndex: capturedAggregate.scales.x.domain.length,
              hasOther: hasOther,
            };
          } else {
            let keys = Array.from(mapTotal.keys()).map((x) => JSON.parse(x));
            keys = keys.sort((a, b) => {
              let sa = typeof a == "string" ? Infinity : a[0];
              let sb = typeof b == "string" ? Infinity : b[0];
              return sa - sb;
            });
            let items = keys.map((d) => ({
              x: d,
              total: mapTotal.get(keyfunc(d)) ?? 0,
              selected: mapSelected.get(keyfunc(d)) ?? 0,
            }));
            let sumTotal = items.reduce((a, b) => a + b.total, 0);
            let sumSelected = items.reduce((a, b) => a + b.selected, 0);
            chartData = {
              items: items,
              sumTotal: sumTotal,
              sumSelected: sumSelected,
              firstSpecialIndex: keys.findIndex((x) => typeof x == "string"),
              hasOther: false,
            };
          }
        }
      });

      // Sync selection with brush
      $effect.pre(() => {
        let clause: any = {
          source: clientSelection,
          clients: new Set([clientSelection]),
          ...(selection != null ? capturedAggregate.clause({ x: selection }) : { value: null, predicate: null }),
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

  const isSame = (a: any, b: any) => JSON.stringify(a) == JSON.stringify(b);

  function toggleSelection(value: string, shift: boolean) {
    if (selection == null || selection.length == 0) {
      selection = [value];
    } else {
      let exists = selection.findIndex((x) => isSame(x, value)) >= 0;
      if (shift) {
        if (exists) {
          selection = selection.filter((x) => !isSame(x, value));
        } else {
          selection = [...selection, value];
        }
      } else {
        if (exists) {
          selection = null;
        } else {
          selection = [value];
        }
      }
    }
  }

  // Sync with state store
  $effect.pre(() =>
    syncState(
      stateStore,
      () => ({
        selection: selection,
        expanded: expanded,
        percentage: percentage,
      }),
      (value) => {
        selection = value.selection;
        expanded = value.expanded;
        percentage = value.percentage;
      },
    ),
  );

  const fmt = format(".6");
  function display(x: string | [number, number]) {
    if (typeof x == "string") {
      return x;
    } else {
      return "[" + fmt(x[0]) + ", " + fmt(x[1]) + ")";
    }
  }

  function formatPercentage(x: number, total: number) {
    if (total == 0) {
      return "-%";
    } else {
      return ((x / total) * 100).toFixed(1) + "%";
    }
  }
</script>

<div class="flex flex-col text-sm w-full select-none" bind:clientWidth={chartWidth}>
  {#if chartData}
    {#each chartData.items as bar, i}
      {@const selected =
        selection == null || selection.length == 0 || selection.findIndex((x) => isSame(x, bar.x)) >= 0}
      {@const hasSelection = !chartData.items.every((x) => x.total == x.selected)}
      {#if i == chartData.firstSpecialIndex}
        <hr class="mt-1 mb-1 border-slate-300 dark:border-slate-500 border-dashed" />
      {/if}
      <button
        class="text-left items-center flex py-0.5"
        onclick={(e) => toggleSelection(bar.x, e.shiftKey)}
        title={bar.x}
      >
        <div class="w-40 flex-none overflow-hidden whitespace-nowrap text-ellipsis pr-1">
          <span class:text-gray-400={!selected} class:dark:text-gray-400={!selected}>{display(bar.x)}</span>
        </div>
        <div class="flex-1 h-4 relative">
          {#if selected}
            {#if !percentage}
              <div
                class="absolute left-0 top-0 bottom-0 rounded-sm"
                style:background={colors.markColorFade}
                style:width="{xScaleAdjusted(bar.total)}px"
              ></div>
            {/if}
            <div
              class="absolute left-0 top-0 bottom-0 rounded-sm"
              style:background={colors.markColor}
              style:width="{xScaleAdjusted(bar.selected)}px"
            ></div>
          {:else}
            {#if !percentage}
              <div
                class="absolute left-0 top-0 bottom-0 rounded-sm"
                style:background={colors.markColorGrayFade}
                style:width="{xScaleAdjusted(bar.total)}px"
              ></div>
            {/if}
            <div
              class="absolute left-0 top-0 bottom-0 rounded-sm"
              style:background={colors.markColorGray}
              style:width="{xScaleAdjusted(bar.selected)}px"
            ></div>
          {/if}
        </div>
        <div class="flex-none">
          <span
            class="text-slate-400 dark:text-slate-500"
            class:!text-gray-200={!selected}
            class:dark:!text-gray-600={!selected}
            title={hasSelection
              ? `${bar.selected.toLocaleString()} / ${bar.total.toLocaleString()} (${formatPercentage(bar.selected, bar.total)})\n${formatPercentage(bar.selected, chartData.sumSelected)} of selection`
              : `${bar.total.toLocaleString()}\n${formatPercentage(bar.total, chartData.sumTotal)} of all rows`}
          >
            {#if hasSelection}
              {#if percentage}
                {formatPercentage(bar.selected, chartData.sumSelected)}
              {:else}
                {bar.selected.toLocaleString() + " / " + bar.total.toLocaleString()}
              {/if}
            {:else if percentage}
              {formatPercentage(bar.total, chartData.sumTotal)}
            {:else}
              {bar.total.toLocaleString()}
            {/if}
          </span>
        </div>
      </button>
    {/each}

    <div class="flex">
      <div class="flex-1 pl-40 mr-2 overflow-hidden">
        {#if expanded || chartData.hasOther}
          <button
            class="py-0.5 text-left text-slate-400 dark:text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 whitespace-nowrap text-ellipsis w-full overflow-hidden"
            onclick={() => {
              expanded = !expanded;
              if (expanded == false) {
                selection = null;
              }
            }}
          >
            {#if expanded}
              ↑ Show up to {MAX_BARS} values
            {:else}
              ↓ Show up to {MAX_BARS_EXPANDED} values
            {/if}
          </button>
        {/if}
      </div>

      <div class="flex">
        <InlineSelect
          options={[
            { value: "true", label: "%" },
            { value: "false", label: "#/#" },
          ]}
          value={percentage}
          onChange={(v) => (percentage = v == "true")}
        />
      </div>
    </div>
  {/if}
</div>
