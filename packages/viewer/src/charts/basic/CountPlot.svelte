<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { makeClient, type Coordinator, type Selection, type SelectionClause } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";

  import InlineSelect from "../../widgets/InlineSelect.svelte";
  import Container from "../common/Container.svelte";
  import CountPlotBar from "./CountPlotBar.svelte";

  import type { ChartViewProps } from "../chart.js";
  import { resolveChartTheme } from "../common/theme.js";
  import type { SQLField } from "../spec/spec.js";
  import type { CountPlotSpec } from "./types.js";

  interface State {
    /** List of selected categories */
    selection?: string[];
  }

  const OTHER_VALUE = "(other)";
  const NULL_VALUE = "(null)";

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
  let theme = $derived(resolveChartTheme($colorScheme, $themeConfig));

  let { selection } = $derived(chartState);
  let { limit = 10, labels = "#/#", order = "total-descending" } = $derived(spec);
  let isListData = $derived(spec.data.isList ?? false);
  let showTotalBars = $derived(labels == "#/#");

  interface Item {
    value: string;
    count: number;
    countSelected: number;
    special?: "null" | "other";
  }

  interface ChartData {
    items: Item[];
    sumSelected: number;
    sumTotal: number;
  }

  let chartData = $state.raw<ChartData | undefined>(undefined);
  let chartWidth = $state.raw(400);

  let maxCount = $derived(
    chartData?.items
      .filter((x) => x.special == null)
      .reduce((a, b) => Math.max(a, showTotalBars ? b.count : b.countSelected), 0) ?? 0,
  );

  function fieldExpr(field: SQLField): SQL.ExprNode {
    if (typeof field == "string") {
      return SQL.column(field);
    } else {
      return SQL.sql`${field.sql}`;
    }
  }

  function filterExprToExpr(filter: SQL.FilterExpr | undefined | null): SQL.ExprNode {
    if (filter == null) {
      return SQL.literal(true);
    } else if (filter instanceof Array) {
      if (filter.length == 0) {
        return SQL.literal(true);
      } else {
        return SQL.and(...filter.map(filterExprToExpr));
      }
    } else {
      if (typeof filter == "string") {
        return SQL.sql`${filter}`;
      } else if (typeof filter == "boolean") {
        return SQL.literal(filter);
      } else {
        return filter;
      }
    }
  }

  function orderExpression(order: CountPlotSpec["order"]): SQL.ExprNode {
    if (order instanceof Array) {
      // Here we just check if the value is in the given list,
      // post-processing code will sort them.
      let literals = order.map((x) => SQL.literal(x));
      return SQL.desc(SQL.isIn("value", literals));
    }
    switch (order ?? "total-descending") {
      case "alphabetical":
        return SQL.asc("value");
      case "selected-ascending":
        return SQL.asc("countSelected");
      case "selected-descending":
        return SQL.desc("countSelected");
      case "total-ascending":
        return SQL.asc("count");
      case "total-descending":
      default:
        return SQL.desc("count");
    }
  }

  function initializeClient(
    coordinator: Coordinator,
    table: string,
    field: SQLField,
    filter: Selection,
    order: CountPlotSpec["order"],
    limit: number,
    isListData: boolean,
  ) {
    if (order instanceof Array) {
      limit = Math.max(limit, order.length);
    }
    let client = makeClient({
      coordinator: coordinator,
      selection: filter,
      query: (predicate) => {
        if (!isListData) {
          let expr = SQL.cast(fieldExpr(field), "TEXT");
          return SQL.Query.from(table)
            .select({
              value: expr,
              count: SQL.count(),
              countSelected: SQL.sum(SQL.cast(filterExprToExpr(predicate), "INT")),
              total: SQL.sql`(${SQL.Query.from(table).select({ count: SQL.count() })})`,
              totalSelected: SQL.sql`(${SQL.Query.from(table).select({ count: SQL.count() }).where(predicate)})`,
            })
            .groupby(expr)
            .orderby(
              SQL.isNotNull(expr), // Make sure the null item is included.
              orderExpression(order),
              SQL.asc("value"),
            )
            .limit(limit + 1);
        } else {
          let intermediateTable = "__count_plot_intermediate__";
          return SQL.Query.with({
            // Intermediate table with unnested values
            [intermediateTable]: SQL.Query.from(table).select({
              value: SQL.sql`UNNEST(${fieldExpr(field)})::TEXT`,
              predicate: SQL.cast(filterExprToExpr(predicate), "INT"),
            }),
          })
            .from(intermediateTable)
            .select({
              value: "value",
              count: SQL.count(),
              countSelected: SQL.sum("predicate"),
              total: SQL.sql`(${SQL.Query.from(intermediateTable).select({ count: SQL.count() })})`,
              totalSelected: SQL.sql`(${SQL.Query.from(intermediateTable).select({ count: SQL.sum("predicate") })})`,
            })
            .groupby("value")
            .orderby(
              SQL.isNotNull("value"), // Make sure the null item is included.
              orderExpression(order),
              SQL.asc("value"),
            )
            .limit(limit + 1);
        }
      },
      queryResult: (result: any) => {
        let rows: {
          value: string | null;
          count: number;
          countSelected: number;
          total: number;
          totalSelected: number;
        }[] = result.toArray();

        if (rows.length == 0) {
          chartData = undefined;
          return;
        }
        let { total, totalSelected } = rows[0];
        if (total == null) {
          total = 0;
        }
        if (totalSelected == null) {
          totalSelected = 0;
        }
        let data: ChartData = {
          items: [],
          sumSelected: totalSelected,
          sumTotal: total,
        };
        for (let row of rows) {
          if (row.value != null && data.items.length < limit) {
            data.items.push({
              value: row.value,
              count: row.count ?? 0,
              countSelected: row.countSelected ?? 0,
            });
          }
        }
        if (order instanceof Array) {
          let sortKey = (item: Item) => {
            let idx = order.indexOf(item.value);
            return idx < 0 ? Infinity : idx;
          };
          data.items.sort((a, b) => sortKey(a) - sortKey(b));
        }
        for (let row of rows) {
          if (row.value == null) {
            data.items.push({
              value: NULL_VALUE,
              count: row.count ?? 0,
              countSelected: row.countSelected ?? 0,
              special: "null",
            });
          }
        }
        let sumCount = data.items.reduce((a, b) => a + b.count, 0);
        let sumCountSelected = data.items.reduce((a, b) => a + b.countSelected, 0);
        if (sumCount < total || sumCountSelected < totalSelected) {
          data.items.push({
            value: OTHER_VALUE,
            count: total - sumCount,
            countSelected: totalSelected - sumCountSelected,
            special: "other",
          });
        }

        chartData = data;
      },
    });

    function makePredicate(selection: string[]): SQL.ExprNode {
      if (selection.length == 0) {
        return SQL.literal(true);
      }
      if (!isListData) {
        // Normal mode, field values are texts
        let expr = SQL.cast(fieldExpr(field), "TEXT");
        return SQL.or(
          ...selection.map((sel) => {
            if (sel == NULL_VALUE) {
              return SQL.isNull(expr);
            } else if (sel == OTHER_VALUE) {
              let literals =
                chartData?.items.filter((x) => x.special == undefined).map((v) => SQL.literal(v.value)) ?? [];
              return SQL.not(SQL.isIn(expr, literals));
            } else {
              return SQL.isNotDistinct(expr, SQL.literal(sel));
            }
          }),
        );
      } else {
        // List mode, field values are lists
        let expr = fieldExpr(field);
        let r = SQL.or(
          ...selection.map((sel) => {
            if (sel == NULL_VALUE) {
              return SQL.listContains(expr, SQL.sql`NULL`);
            } else if (sel == OTHER_VALUE) {
              // List contains something not in the list of known values
              let literals =
                chartData?.items.filter((x) => x.special == undefined).map((v) => SQL.literal(v.value)) ?? [];
              return SQL.sql`len(list_filter(${expr}, x -> x NOT IN (${literals.join(",")}))) > 0`;
            } else {
              return SQL.listContains(expr, SQL.literal(sel));
            }
          }),
        );
        return r;
      }
    }

    let source = {
      reset: () => {
        onStateChange({ selection: undefined });
      },
    };

    // Sync selection with brush
    $effect.pre(() => {
      let clause: SelectionClause = {
        source: source,
        clients: new Set([client]),
        ...(selection != null && selection.length > 0
          ? { value: selection, predicate: makePredicate(selection) }
          : { value: null, predicate: null }),
      };
      filter.update(clause);
    });

    return () => {
      client.destroy();
      filter.update({
        source: source,
        clients: new Set([client]),
        value: null,
        predicate: null,
      });
    };
  }

  $effect.pre(() => {
    return initializeClient(coordinator, context.table, spec.data.field, context.filter, order, limit, isListData);
  });

  function toggleSelection(value: string, shift: boolean) {
    if (selection == undefined || selection.length == 0) {
      onStateChange({ selection: [value] });
    } else {
      let exists = selection.findIndex((x) => x == value) >= 0;
      if (shift) {
        if (exists) {
          onStateChange({ selection: selection.filter((x) => x != value) });
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

  function formatPercentage(x: number, total: number) {
    if (total == 0) {
      return "-%";
    } else {
      return ((x / total) * 100).toFixed(1) + "%";
    }
  }

  function formatItem(
    item: Item,
    chartData: ChartData,
    labels: CountPlotSpec["labels"],
    hasSelection: boolean,
  ): { label: string; title: string } {
    let label: string;
    let title: string[];

    switch (labels ?? "#/#") {
      case "#":
        label = hasSelection ? item.countSelected.toLocaleString() : item.count.toLocaleString();
        break;
      case "%":
        label = hasSelection
          ? formatPercentage(item.countSelected, chartData.sumSelected)
          : formatPercentage(item.count, chartData.sumTotal);
        break;
      case "#/#":
        label = hasSelection
          ? item.countSelected.toLocaleString() + " / " + item.count.toLocaleString()
          : item.count.toLocaleString();
        break;
      default:
        label = "";
    }

    if (!isListData) {
      title = hasSelection
        ? [
            `${item.countSelected.toLocaleString()} / ${item.count.toLocaleString()} (${formatPercentage(item.countSelected, item.count)})`,
            `${formatPercentage(item.countSelected, chartData.sumSelected)} of selection`,
          ]
        : [`${item.count.toLocaleString()}`, `${formatPercentage(item.count, chartData.sumTotal)} of all rows`];
    } else {
      title = hasSelection
        ? [
            `${item.countSelected.toLocaleString()} / ${item.count.toLocaleString()} (${formatPercentage(item.countSelected, item.count)})`,
            `${formatPercentage(item.countSelected, chartData.sumSelected)} of selection`,
            `(Occurrences in the list values)`,
          ]
        : [
            `${item.count.toLocaleString()}`,
            `${formatPercentage(item.count, chartData.sumTotal)} of all occurrences`,
            `(Occurrences in the list values)`,
          ];
    }

    return { label: label, title: title.join("\n") };
  }
</script>

<Container width={width} height={height} scrollY={true}>
  <div class="flex flex-col text-sm w-full select-none" bind:clientWidth={chartWidth}>
    {#if chartData}
      {@const firstSpecialIndex = chartData.items.findIndex((x) => x.special != undefined)}
      {#each chartData.items as bar, i}
        {@const selected =
          selection == undefined || selection.length == 0 || selection.findIndex((x) => x == bar.value) >= 0}
        {@const hasSelection = !chartData.items.every((x) => x.count == x.countSelected)}
        {@const formatted = formatItem(bar, chartData, labels, hasSelection)}
        {#if i == firstSpecialIndex}
          <hr class="mt-1 mb-1 border-slate-300 dark:border-slate-500 border-dashed" />
        {/if}
        <button
          class="text-left items-center flex py-0.5"
          onclick={(e) => toggleSelection(bar.value, e.shiftKey)}
          title={bar.value}
        >
          <div class="w-40 flex-none overflow-hidden whitespace-nowrap text-ellipsis pr-1">
            <span class:text-gray-400={!selected} class:dark:text-gray-400={!selected}>{bar.value}</span>
          </div>
          <CountPlotBar
            selected={selected}
            bars={selected
              ? [
                  ...(showTotalBars ? [{ value: bar.count, color: theme.markColorFade }] : []),
                  { value: bar.countSelected, color: theme.markColor },
                ]
              : [
                  ...(showTotalBars ? [{ value: bar.count, color: theme.markColorGrayFade }] : []),
                  { value: bar.countSelected, color: theme.markColorGray },
                ]}
            maxValue={maxCount}
            width={chartWidth - 250}
            label={formatted.label}
            title={formatted.title}
          />
        </button>
      {/each}

      <div class="flex mt-0.5">
        <div class="w-40 flex-none overflow-hidden whitespace-nowrap text-ellipsis pr-1">
          {#if isListData}
            <div class="py-0.5 text-slate-400 dark:text-slate-500">(Occurrences in lists)</div>
          {/if}
        </div>
        <div class="flex-1 mr-2 overflow-hidden">
          {#if limit != 10 || chartData.items.findIndex((x) => x.special == "other") >= 0}
            <button
              class="py-0.5 text-left text-slate-400 dark:text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 whitespace-nowrap text-ellipsis w-full overflow-hidden"
              onclick={() => {
                let newLimit = limit < 50 ? 100 : 10;
                onSpecChange({ limit: newLimit });
                if (newLimit < limit) {
                  onStateChange({ selection: undefined });
                }
              }}
            >
              {#if limit < 50}
                ↓ Up to 100 values
              {:else}
                ↑ Up to 10 values
              {/if}
            </button>
          {/if}
        </div>

        <div class="flex gap-1">
          {#if typeof order == "string"}
            <InlineSelect
              options={[
                { value: "total-descending", label: "↓ Total" },
                { value: "selected-descending", label: "↓ Selected" },
                { value: "total-ascending", label: "↑ Total" },
                { value: "selected-ascending", label: "↑ Selected" },
                { value: "alphabetical", label: "↓ A-Z" },
              ]}
              value={order ?? "total-descending"}
              onChange={(v) => onSpecChange({ order: v })}
            />
          {/if}
          <InlineSelect
            options={[
              { value: "%", label: "%" },
              { value: "#", label: "#" },
              { value: "#/#", label: "#/#" },
            ]}
            value={labels ?? "#/#"}
            onChange={(v) => onSpecChange({ labels: v })}
          />
        </div>
      </div>
    {/if}
  </div>
</Container>
