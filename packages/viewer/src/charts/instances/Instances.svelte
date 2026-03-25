<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { deepMemo } from "@embedding-atlas/utils";
  import { makeClient } from "@uwdata/mosaic-core";
  import * as SQL from "@uwdata/mosaic-sql";
  import { untrack } from "svelte";

  import PaginatorControls from "../../widgets/PaginatorControls.svelte";
  import SegmentedControl from "../../widgets/SegmentedControl.svelte";
  import Cards from "./Cards.svelte";
  import SortOrderControl from "./SortOrderControl.svelte";
  import Table from "./Table.svelte";

  import { IconCardView, IconRight, IconTableView } from "../../assets/icons.js";
  import type { ColumnStyle } from "../../renderers/types.js";
  import { isolatedWritable } from "../../utils/store.js";
  import type { ChartViewProps, RowID } from "../chart.js";
  import { instancesQuery } from "./query.js";
  import type { InstancesSpec, InstancesState, SortOrder } from "./types.js";

  let {
    context,
    spec,
    state: chartState,
    height,
    onSpecChange,
    onStateChange,
  }: ChartViewProps<InstancesSpec, InstancesState> = $props();

  // svelte-ignore state_referenced_locally
  let { columnStyles: contextColumnStyles } = context;

  // Merge spec columnStyles with global ones (spec takes precedence)
  let columnStyles = $derived({ ...$contextColumnStyles, ...spec.columnStyles });

  // svelte-ignore state_referenced_locally
  let highlight = context.highlight;
  let isolatedHighlight = isolatedWritable(highlight);

  let viewMode = $derived((spec.viewMode ?? "table") as "table" | "cards");
  let offset = $derived(chartState.offset ?? 0);
  let pageSize = $derived(spec.pageSize ?? 100);

  let contentView = $state.raw<Table | Cards | undefined>(undefined);
  let viewContainer = $state.raw<HTMLElement | undefined>(undefined);

  // Column widths (local state, not persisted)
  let defaultColumnWidths = $state.raw<Record<string, number>>({});

  // Subscribe to highlight changes
  $effect.pre(() => {
    let isOnMount = true;
    let previousValue: RowID[] | null = null;
    return isolatedHighlight.subscribe((v) => {
      // Don't animate immediately on mount.
      if (isOnMount) {
        isOnMount = false;
        previousValue = v;
        return;
      }
      // Animate when a single new point is added.
      let newIDs = v ?? [];
      let oldIDs = previousValue ?? [];
      let enteringIDs = newIDs.filter((x) => oldIDs.indexOf(x) < 0);
      if (enteringIDs.length == 1) {
        animateToPoint(enteringIDs[0]);
      }
      previousValue = v;
    });
  });

  interface Data {
    data: Record<string, any>[];
    columns: string[];
    offset: number;

    offsetForId?: (id: RowID) => Promise<number | undefined>;
  }

  // Data loading
  let totalCount = $state.raw(0);
  let data = $state.raw<Data | undefined>(undefined);

  // Derive current page and page count for PaginatorControls
  let currentPage = $derived(Math.floor(offset / pageSize));
  let pageCount = $derived(Math.ceil(totalCount / pageSize));

  // Reset the offset and scroll to top.
  function resetOffset() {
    untrack(() => {
      if (offset != 0) {
        onStateChange({ offset: 0 });
      }
      if (viewContainer) {
        viewContainer.scrollTop = 0;
      }
    });
  }

  function createClients(options: {
    query?: string;
    columns?: string[];
    columnStyles: Record<string, ColumnStyle>;
    sort?: SortOrder;
    pageSize: number;
  }) {
    let isOriginalTable = options.query == undefined;
    let baseQuery = (predicate?: SQL.FilterExpr | null) =>
      instancesQuery({ query: options.query, table: context.table, predicate: predicate });

    // Build orderby expressions from sort specification
    let orderByExprs = (options.sort ?? []).map((s) => {
      let col = SQL.column(s.column);
      return s.direction === "descending" ? SQL.desc(col) : SQL.asc(col);
    });

    let columnNames: string[] = [];
    let lastQueryOffset = 0;
    let lastQueryPredicate: SQL.FilterExpr | undefined = undefined;

    let clientTotal = makeClient({
      coordinator: context.coordinator,
      selection: context.filter,
      query: (predicate) => {
        return SQL.Query.from(baseQuery(predicate)).select({ count: SQL.count() });
      },
      queryResult: (result: any) => {
        totalCount = result.get(0).count;
      },
    });

    let client = makeClient({
      coordinator: context.coordinator,
      selection: context.filter,
      prepare: async () => {
        let desc = await context.coordinator.query(SQL.Query.describe(baseQuery()));
        columnNames = desc.toArray().map((x) => x.column_name);
        if (options.columns) {
          let specifiedColumns = new Set(options.columns);
          columnNames = columnNames.filter((x) => specifiedColumns.has(x));
        }
        // Filter out hidden columns
        columnNames = columnNames.filter((col) => options.columnStyles[col]?.display !== "hidden");

        // Get sample data for column widths
        let widthQuery = SQL.Query.from(baseQuery())
          .select(
            Object.fromEntries([
              ...(isOriginalTable ? [["__id__", SQL.column(context.id)]] : []),
              ...columnNames.map((x) => [x, SQL.column(x)]),
            ]),
          )
          .limit(10)
          .offset(0);
        let widthResult = await context.coordinator.query(widthQuery);
        let sampleData = widthResult.toArray();
        defaultColumnWidths = Object.fromEntries(
          columnNames.map((col) => [
            col,
            sampleData.reduce(
              (max: number, row: any) => Math.max(max, widthForContent(row[col])),
              widthForContent(col), // Also take column name into account
            ),
          ]),
        );
      },
      query: (predicate) => {
        lastQueryOffset = offset;
        lastQueryPredicate = predicate;
        return SQL.Query.from(baseQuery(predicate))
          .select(
            Object.fromEntries([
              ...(isOriginalTable ? [["__id__", SQL.column(context.id)]] : []),
              ...columnNames.map((x) => [x, SQL.column(x)]),
            ]),
          )
          .orderby(orderByExprs)
          .limit(options.pageSize)
          .offset(offset);
      },
      queryResult: (result: any) => {
        data = {
          data: result.toArray(),
          columns: columnNames,
          offset: lastQueryOffset,
          offsetForId: isOriginalTable
            ? async (id) => {
                // Build ROW_NUMBER window function with same sort order as main query
                let idOffset = SQL.Query.from(baseQuery(lastQueryPredicate)).select({
                  id: SQL.column(context.id),
                  offset: orderByExprs.length > 0 ? SQL.row_number().orderby(...orderByExprs) : SQL.row_number(),
                });
                let query = SQL.Query.from(idOffset)
                  .select({ offset: SQL.column("offset") })
                  .where(SQL.eq(SQL.column("id"), SQL.literal(id)));
                let result = await context.coordinator.query(query);
                return result.get(0)?.offset;
              }
            : undefined,
        };
      },
    });

    $effect.pre(() => {
      // When offset changes, rerun the query.
      if (offset != lastQueryOffset) {
        client.requestQuery();
      }
    });

    return () => {
      clientTotal.destroy();
      client.destroy();
    };
  }

  // Reset offset and create a new client when critical parts of the spec change
  let clientsParams = $derived.by(
    deepMemo(() => ({
      query: spec.query,
      columns: spec.columns,
      columnStyles: columnStyles,
      sort: spec.sort,
      pageSize: pageSize,
    })),
  );

  $effect.pre(() => {
    resetOffset();
    return createClients(clientsParams);
  });

  // Reset offset when predicate changes
  $effect.pre(() => {
    let callback = () => {
      resetOffset();
    };
    context.filter.addEventListener("value", callback);
    return () => {
      context.filter.removeEventListener("value", callback);
    };
  });

  // Calculate width based on content length
  function widthForContent(content: any): number {
    let characterLength = String(content).length;
    return Math.min(600, Math.max(80, characterLength * 8 + 40));
  }

  // Animate to a point. When the point is in the same page, scroll to the point;
  // otherwise, go to the page with the point, and reveal the element directly.
  async function animateToPoint(id: RowID) {
    if (spec.query != null) {
      // For custom queries we do not animate.
      return;
    }
    if (data == null) {
      return;
    }

    // Check if highlighted item is in current page
    let isInCurrentPage = data.data.some((row) => row.__id__ === id);
    if (isInCurrentPage) {
      contentView?.getElementForId(id)?.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      let newOffset = await data?.offsetForId?.(id);
      if (newOffset != undefined) {
        // Make sure it's a multiple of page number.
        newOffset = Math.floor(newOffset / pageSize) * pageSize;
        scrollToOnLoadPage = { offset: newOffset, id: id };
        onStateChange({ offset: newOffset });
      }
    }
  }

  let scrollToOnLoadPage = $state.raw<{ offset: number; id: RowID } | undefined>(undefined);

  // Helper effect for animateToPoint, to show the new point when switching to a new page.
  $effect(() => {
    if (!scrollToOnLoadPage) {
      return;
    }
    let scrollTo = scrollToOnLoadPage;
    let currentData = data;
    if (currentData?.offset == scrollTo.offset) {
      scrollToOnLoadPage = undefined;
      untrack(() => {
        contentView?.getElementForId(scrollTo.id)?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    }
  });

  function handlePageChange(page: number) {
    onStateChange({ offset: page * pageSize });
  }

  function handleLoadNext() {
    onStateChange({ offset: Math.min(totalCount - 1, offset + pageSize) });
  }

  function handleRowClick(rowId: RowID | null | undefined, event: MouseEvent) {
    if (rowId == null) {
      return;
    }
    isolatedHighlight.update((value) => {
      if (event.shiftKey || event.ctrlKey || event.metaKey) {
        if (value == null) {
          return [rowId];
        }
        if (value.indexOf(rowId) >= 0) {
          return value.filter((x) => x != rowId);
        } else {
          return [...value, rowId];
        }
      } else {
        if (value != null && value.length == 1 && value.indexOf(rowId) >= 0) {
          return null;
        } else {
          return [rowId];
        }
      }
    });
  }
</script>

<div
  class="w-full flex flex-col overflow-hidden rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
  style:height={`${height ?? spec.defaultHeight ?? 500}px`}
>
  <div class="flex items-center justify-between px-2 py-0.5 border-b border-slate-200 dark:border-slate-700 gap-4">
    <div class="flex items-center gap-4 flex-shrink-0">
      <SegmentedControl
        value={viewMode}
        onChange={(v) => onSpecChange({ viewMode: v as "table" | "cards" })}
        options={[
          { value: "table", icon: IconTableView, title: "Table view" },
          { value: "cards", icon: IconCardView, title: "Card view" },
        ]}
      />
      <PaginatorControls currentPage={currentPage} pageCount={pageCount} onChange={handlePageChange} />
      <SortOrderControl value={spec.sort} onChange={(value) => onSpecChange({ sort: value })} />
    </div>
  </div>

  <div class="flex-1 min-h-0 overflow-auto" bind:this={viewContainer}>
    {#if data != null}
      {#if viewMode === "table"}
        <Table
          bind:this={contentView}
          data={data.data}
          columns={data.columns}
          columnDescs={context.columns}
          columnStyles={columnStyles}
          defaultColumnWidths={defaultColumnWidths}
          highlight={$highlight}
          sort={spec.sort}
          onRowClick={handleRowClick}
          onSortChange={(value) => onSpecChange({ sort: value })}
        />

        {#if offset + pageSize < totalCount}
          <div class="p-3 flex justify-center">
            <button class="px-4 py-2 text-sm flex items-center gap-1" onclick={handleLoadNext}>
              Next Page
              <IconRight />
            </button>
          </div>
        {/if}
      {:else}
        <Cards
          bind:this={contentView}
          data={data.data}
          columns={data.columns}
          columnStyles={columnStyles}
          highlight={$highlight}
          cardTemplate={spec.cardTemplate}
          onRowClick={handleRowClick}
        />

        {#if offset + pageSize < totalCount}
          <div class="p-3 flex justify-center">
            <button class="px-4 py-2 text-sm flex items-center gap-1" onclick={handleLoadNext}>
              Next Page
              <IconRight />
            </button>
          </div>
        {/if}
      {/if}
    {:else}
      <div class="flex items-center justify-center h-full">
        <div class="text-slate-500 dark:text-slate-400">Loading...</div>
      </div>
    {/if}
  </div>
</div>
