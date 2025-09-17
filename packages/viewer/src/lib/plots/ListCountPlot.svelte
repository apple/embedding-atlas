<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  import type { Coordinator } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  async function listStats(
    coordinator: Coordinator,
    table: string,
    field: string,
    limit: number,
  ): Promise<{
    values: { value: string; count: number }[];
    hasOther: boolean;
  }> {
    let column = SQL.column(field, table);
    let result = await coordinator.query(
      SQL.Query.from(
        SQL.Query.from(table).select({
          value: SQL.sql`UNNEST(${column})`,
        }),
      )
        .select({
          value: "value",
          count: SQL.count(),
        })
        .groupby("value")
        .orderby(SQL.desc("count"))
        .limit(limit + 1),
    );
    let values = Array.from(result) as { value: string; count: number }[];
    return { values: values.slice(0, limit), hasOther: values.length > limit };
  }

  function makePredicate(field: string, selection: string[]) {
    return SQL.or(...selection.map((v) => SQL.sql`${SQL.literal(v)} IN ${SQL.column(field)}`));
  }
</script>

<script lang="ts">
  import { type Selection, makeClient } from "@uwdata/mosaic-core";
  import { format } from "d3-format";
  import { scaleLinear } from "d3-scale";

  import { Context } from "../contexts.js";
  import { plotColors } from "./colors.js";

  import type { PlotStateStore } from "./plot_state_store.js";
  import { syncState } from "./utils.svelte";

  const MAX_BARS = 10;
  const MAX_BARS_EXPANDED = 100;

  interface Props {
    table: string;
    field: string;
    filter: Selection;
    stateStore?: PlotStateStore<{ selection: string[] | null; expanded: boolean }>;
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
    firstSpecialIndex: number;
    hasOther: boolean;
  }

  let selection: string[] | null = $state.raw(null);
  let expanded = $state.raw(false);

  let chartData = $state.raw<ChartData | null>(null);
  let chartWidth = $state.raw(400);

  let maxCount = $derived(chartData?.items.reduce((a, b) => Math.max(a, b.total), 0) ?? 0);
  let xScale = $derived(scaleLinear([0, Math.max(1, maxCount)], [0, chartWidth - 250]));

  // Adjust scale so the minimum width for non-zero count is 1px.
  let xScaleAdjusted = $derived((v: number) => (v != 0 ? Math.max(1, xScale(v)) : 0));

  let colors = $derived($darkMode ? plotColors.dark : plotColors.light);

  function initializeClient(coordinator: Coordinator, table: string, field: string, filter: Selection, limit: number) {
    let stats: any | null = $state.raw(null);

    // Query the stats
    listStats(coordinator, table, field, limit).then((r) => {
      stats = r;
    });

    function createClient(selection: Selection | null, values: string[], callback: (bins: any[]) => void) {
      return makeClient({
        coordinator: coordinator,
        selection: selection ?? undefined,
        query: (predicate) => {
          let column = SQL.column(field, table);
          return SQL.Query.from(
            SQL.Query.from(table)
              .select({
                value: SQL.sql`UNNEST(${column})`,
              })
              .where(predicate),
          )
            .select({
              x: "value",
              count: SQL.count(),
            })
            .where(
              SQL.isIn(
                "value",
                values.map((x) => SQL.literal(x)),
              ),
            )
            .groupby("value")
            .orderby(SQL.desc("count"));
        },
        queryResult: (data: any) => {
          callback(Array.from(data));
        },
      });
    }

    $effect.pre(() => {
      if (stats == null) {
        return;
      }

      let xDomain: string[] = stats.values.map((x: any) => x.value);
      let hasOther = stats.hasOther;

      let allItems: Bin[] = $state.raw([]);
      let filteredItems: Bin[] = $state.raw([]);

      let clientBase = createClient(null, xDomain, (data) => {
        allItems = data;
      });
      let clientSelection = createClient(filter, xDomain, (data) => {
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
          let items = xDomain.map((d) => ({
            x: d,
            total: mapTotal.get(keyfunc(d)) ?? 0,
            selected: mapSelected.get(keyfunc(d)) ?? 0,
          }));
          chartData = {
            items: items,
            firstSpecialIndex: xDomain.length,
            hasOther: hasOther,
          };
        }
      });

      // Sync selection with brush
      $effect.pre(() => {
        let clause: any = {
          source: clientSelection,
          clients: new Set([clientSelection]),
          ...(selection != null
            ? { value: selection, predicate: makePredicate(field, selection) }
            : { value: null, predicate: null }),
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
    initializeClient(coordinator, table, field, filter, expanded ? MAX_BARS_EXPANDED : MAX_BARS);
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
      }),
      (value) => {
        selection = value.selection;
        expanded = value.expanded;
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
            <div
              class="absolute left-0 top-0 bottom-0 rounded-sm"
              style:background={colors.markColorFade}
              style:width="{xScaleAdjusted(bar.total)}px"
            ></div>

            <div
              class="absolute left-0 top-0 bottom-0 rounded-sm"
              style:background={colors.markColor}
              style:width="{xScaleAdjusted(bar.selected)}px"
            ></div>
          {:else}
            <div
              class="absolute left-0 top-0 bottom-0 rounded-sm"
              style:background={colors.markColorGrayFade}
              style:width="{xScaleAdjusted(bar.total)}px"
            ></div>
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
              ? `${bar.total.toLocaleString()} rows contain "${bar.x}"; ${bar.selected.toLocaleString()} (${formatPercentage(bar.selected, bar.total)}) in selection`
              : `${bar.total.toLocaleString()} rows contain "${bar.x}"`}
          >
            {#if hasSelection}
              {bar.selected.toLocaleString() + " / " + bar.total.toLocaleString()}
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
    </div>
  {/if}
</div>
