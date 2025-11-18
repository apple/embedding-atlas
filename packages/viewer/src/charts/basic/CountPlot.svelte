<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { makeClient, type Coordinator, type Selection, type SelectionClause } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import * as d3 from "d3";

  import InlineSelect from "../../widgets/InlineSelect.svelte";
  import Container from "../common/Container.svelte";

  import type { ChartViewProps } from "../chart.js";
  import { computeFieldStats, inferAggregate, type AggregateInfo, type FieldStats } from "../common/aggregate.js";
  import { resolveChartTheme } from "../common/theme.js";
  import type { CountPlotSpec } from "./types.js";

  interface State {
    selection?: string[];
  }

  const MAX_BARS = 10;
  const MAX_BARS_EXPANDED = 100;

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onStateChange,
    onSpecChange,
  }: ChartViewProps<CountPlotSpec, State> = $props();
  let { coordinator, colorScheme, theme: themeConfig } = context;
  let { selection } = $derived(chartState);
  let { expanded, percentage } = $derived(spec);

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

  let chartData = $state.raw<ChartData | undefined>(undefined);
  let chartWidth = $state.raw(400);

  let maxCount = $derived(chartData?.items.reduce((a, b) => Math.max(a, percentage ? b.selected : b.total), 0) ?? 0);
  let xScale = $derived(d3.scaleLinear([0, Math.max(1, maxCount)], [0, chartWidth - 250]));

  // Adjust scale so the minimum width for non-zero count is 1px.
  let xScaleAdjusted = $derived((v: number) => (v != 0 ? Math.max(1, xScale(v)) : 0));

  let theme = $derived(resolveChartTheme($colorScheme, $themeConfig));

  function initializeClient(coordinator: Coordinator, table: string, field: string, filter: Selection) {
    let stats: FieldStats | undefined = $state.raw(undefined);

    // Query the stats
    computeFieldStats(coordinator, SQL.sql`${table}`, SQL.column(field)).then((r) => {
      stats = r;
    });

    // Infer binning from stats
    let aggregate: AggregateInfo | undefined = $derived(
      stats ? inferAggregate({ stats, binCount: expanded ? MAX_BARS_EXPANDED : MAX_BARS }) : undefined,
    );

    function createClient(aggregate: AggregateInfo, selection: Selection | undefined, callback: (bins: any[]) => void) {
      return makeClient({
        coordinator: coordinator,
        selection: selection,
        query: (predicate) => {
          return SQL.Query.from(table)
            .select({ x: aggregate.select, count: SQL.count() })
            .where(predicate)
            .groupby(aggregate.select);
        },
        queryResult: (data: any) => {
          let items: any[] = Array.from(data);
          callback(
            items.map(({ x, count }) => ({
              x: aggregate.field(x),
              count: count,
            })),
          );
        },
      });
    }

    $effect.pre(() => {
      if (aggregate == undefined) {
        return;
      }
      let capturedAggregate = aggregate;

      let allItems: Bin[] = $state.raw([]);
      let filteredItems: Bin[] = $state.raw([]);

      let clientBase = createClient(capturedAggregate, undefined, (data) => {
        allItems = data;
      });
      let clientSelection = createClient(capturedAggregate, filter, (data) => {
        filteredItems = data;
      });
      let source = {
        reset: () => {
          onStateChange({ selection: undefined });
        },
      };

      $effect.pre(() => {
        if (allItems.length > 0) {
          let keyfunc = (x: any) => JSON.stringify(x);
          let mapTotal = new Map<string, number>(allItems.map(({ x, count }) => [keyfunc(x), count]));
          let mapSelected = new Map<string, number>(filteredItems.map(({ x, count }) => [keyfunc(x), count]));

          if (allItems.every((d) => typeof d.x == "string")) {
            let specialValues = capturedAggregate.scale.specialValues ?? [];
            let hasOther = specialValues.filter((x) => x != "(null)").length > 0;
            let items = [...capturedAggregate.scale.domain, ...specialValues].map((d) => ({
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
              firstSpecialIndex: capturedAggregate.scale.domain.length,
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
        let clause: SelectionClause = {
          source: source,
          clients: new Set([clientSelection]),
          ...(selection != undefined
            ? { value: selection, predicate: capturedAggregate.predicate(selection) ?? null }
            : { value: null, predicate: null }),
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

  const isSame = (a: any, b: any) => JSON.stringify(a) == JSON.stringify(b);

  function toggleSelection(value: string, shift: boolean) {
    if (selection == undefined || selection.length == 0) {
      onStateChange({ selection: [value] });
    } else {
      let exists = selection.findIndex((x) => isSame(x, value)) >= 0;
      if (shift) {
        if (exists) {
          onStateChange({ selection: selection.filter((x) => !isSame(x, value)) });
        } else {
          onStateChange({ selection: [...selection, value] });
        }
      } else {
        if (exists) {
          onStateChange({ selection: undefined });
        } else {
          onStateChange({ selection: [value] });
        }
      }
    }
  }

  const fmt = d3.format(".6");
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

<Container width={width} height={height} scrollY={true}>
  <div class="flex flex-col text-sm w-full select-none" bind:clientWidth={chartWidth}>
    {#if chartData}
      {#each chartData.items as bar, i}
        {@const selected =
          selection == undefined || selection.length == 0 || selection.findIndex((x) => isSame(x, bar.x)) >= 0}
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
                  style:background={theme.markColorFade}
                  style:width="{xScaleAdjusted(bar.total)}px"
                ></div>
              {/if}
              <div
                class="absolute left-0 top-0 bottom-0 rounded-sm"
                style:background={theme.markColor}
                style:width="{xScaleAdjusted(bar.selected)}px"
              ></div>
            {:else}
              {#if !percentage}
                <div
                  class="absolute left-0 top-0 bottom-0 rounded-sm"
                  style:background={theme.markColorGrayFade}
                  style:width="{xScaleAdjusted(bar.total)}px"
                ></div>
              {/if}
              <div
                class="absolute left-0 top-0 bottom-0 rounded-sm"
                style:background={theme.markColorGray}
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
                if (expanded == true) {
                  onSpecChange({ expanded: false });
                  onStateChange({ selection: undefined });
                } else {
                  onSpecChange({ expanded: true });
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
            value={percentage?.toString() ?? "false"}
            onChange={(v) => onSpecChange({ percentage: v == "true" })}
          />
        </div>
      </div>
    {/if}
  </div>
</Container>
