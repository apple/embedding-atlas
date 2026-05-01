<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { interactionHandler, type CursorValue } from "@embedding-atlas/utils";

  import ContentRenderer from "../../renderers/ContentRenderer.svelte";

  import { IconSortDown, IconSortUp, IconSortUpDown } from "../../assets/icons.js";

  import type { ColumnStyle } from "../../renderers/types.js";
  import type { ColumnDesc } from "../../utils/database.js";
  import type { RowID } from "../chart.js";
  import { inferColumnFormatters } from "./infer_formatters.js";
  import type { SortOrder } from "./types.js";

  interface Props {
    data: Record<string, any>[];
    columns: string[];
    columnDescs: ColumnDesc[];
    columnStyles: Record<string, ColumnStyle>;
    defaultColumnWidths: Record<string, number>;
    highlight: RowID[] | null;
    sort?: SortOrder;
    onRowClick: (rowId: RowID | null | undefined, event: MouseEvent) => void;
    onSortChange: (sort: SortOrder | undefined) => void;
  }

  let {
    data,
    columns,
    columnDescs,
    columnStyles,
    defaultColumnWidths,
    highlight,
    sort,
    onRowClick,
    onSortChange,
  }: Props = $props();

  let highlightSet = $derived(new Set(highlight));
  let columnFormatters = $derived(inferColumnFormatters(data, columns));

  let idMapper = new Map<RowID, Element>();

  // Table UI state (owned by Table component)
  let columnWidths = $state.raw<Record<string, number>>({});

  let expandedRows = $state.raw<Set<number>>(new Set());
  let hoveredCell = $state.raw<{ row: number; col: string } | null>(null);

  // Get text alignment class based on column type from schema
  function getAlignment(column: string): string {
    const columnDesc = columnDescs.find((col) => col.name === column);
    if (columnDesc?.jsType === "number") {
      return "text-right";
    }
    return "text-left";
  }

  function toggleRowExpansion(rowIndex: number) {
    expandedRows = new Set(
      expandedRows.has(rowIndex) ? [...expandedRows].filter((i) => i !== rowIndex) : [...expandedRows, rowIndex],
    );
  }

  // Column resizing handlers
  function handleResizeStart(column: string) {
    return (e1: CursorValue) => {
      let startWidth = columnWidths[column] ?? defaultColumnWidths[column] ?? 150;
      return {
        move: (e2: CursorValue) => {
          let newWidth = startWidth + e2.clientX - e1.clientX;
          newWidth = Math.max(60, newWidth);
          columnWidths = { ...columnWidths, [column]: newWidth };
        },
      };
    };
  }

  function columnSortOrder(
    column: string,
    sort: SortOrder | undefined,
  ): { order: "ascending" | "descending" | undefined; isPrimary: boolean } | undefined {
    let index = sort?.findIndex((x) => x.column == column) ?? -1;
    if (index < 0) {
      return;
    }
    return { order: sort?.[index].direction, isPrimary: index == 0 };
  }

  function changeColumnSortOrder(column: string, order: "ascending" | "descending" | undefined) {
    let newSort: SortOrder = [];
    if (order != undefined) {
      newSort = [{ column: column, direction: order }, ...(sort?.filter((x) => x.column != column) ?? [])];
    } else {
      newSort = sort?.filter((x) => x.column != column) ?? [];
    }
    onSortChange(newSort.length == 0 ? undefined : newSort);
  }

  export function getElementForId(id: RowID): Element | undefined {
    return idMapper.get(id);
  }
</script>

<table class="border-separate border-spacing-0 table-fixed w-full">
  <thead class="sticky top-0 z-10 bg-white dark:bg-black">
    <tr>
      {#each columns as column}
        {@const sortOrder = columnSortOrder(column, sort)}
        {@const sortButtonHighlight = sortOrder?.isPrimary ?? false}
        {@const width = columnWidths[column] ?? defaultColumnWidths[column] ?? 150}
        <th
          class="px-4 py-1.5 font-normal text-slate-400 dark:text-slate-500 border-b border-slate-200 dark:border-slate-800 whitespace-nowrap relative group {getAlignment(
            column,
          )}"
          style:width="{width}px"
        >
          <div class="flex gap-2 items-center">
            <div class="flex-1 truncate">{column}</div>
            <button
              onclick={() =>
                changeColumnSortOrder(
                  column,
                  sortOrder == undefined ? "ascending" : sortOrder.order == "ascending" ? "descending" : undefined,
                )}
            >
              <div
                class:text-slate-300={!sortButtonHighlight}
                class:dark:text-slate-600={!sortButtonHighlight}
                class:text-slate-600={sortButtonHighlight}
                class:dark:text-slate-200={sortButtonHighlight}
              >
                {#if sortOrder?.order == "ascending"}
                  <IconSortUp />
                {:else if sortOrder?.order == "descending"}
                  <IconSortDown />
                {:else}
                  <IconSortUpDown />
                {/if}
              </div>
            </button>
          </div>
          <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
          <div
            class="absolute -right-[5px] top-0.5 bottom-0.5 w-[12px] cursor-col-resize flex items-center justify-center z-20"
            use:interactionHandler={{ drag: handleResizeStart(column) }}
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize column"
          >
            <div class="w-[2px] h-5 bg-slate-400 dark:bg-slate-500 opacity-20 rounded-sm"></div>
          </div>
        </th>
      {/each}
      <th class="border-b border-slate-200 dark:border-slate-800"></th>
    </tr>
  </thead>
  <tbody>
    {#each data as row, index}
      {@const rowId = row.__id__}
      <tr
        class="transition-colors {highlightSet.has(rowId)
          ? 'bg-blue-100 dark:bg-blue-950'
          : index % 2 === 0
            ? 'bg-white dark:bg-black hover:bg-blue-50 dark:hover:bg-blue-950'
            : 'bg-slate-50 dark:bg-slate-900 hover:bg-blue-50 dark:hover:bg-blue-950'}"
        onclick={(e) => onRowClick(rowId, e)}
        onmousedown={(e) => {
          // Prevent text selection when shift-click to multi-select
          if (e.shiftKey || e.ctrlKey || e.metaKey) {
            e.preventDefault();
          }
        }}
        bind:this={() => idMapper.get(rowId), (v) => idMapper.set(rowId, v)}
      >
        {#each columns as column}
          <td
            class="px-4 py-1.5 text-slate-500 dark:text-slate-400 align-top overflow-hidden relative {getAlignment(
              column,
            )}"
            onmouseenter={() => (hoveredCell = { row: index, col: column })}
            onmouseleave={() => (hoveredCell = null)}
          >
            <div class="wrap-anywhere" class:line-clamp-3={!expandedRows.has(index)}>
              <ContentRenderer value={row[column]} style={columnStyles[column]} formatter={columnFormatters[column]} />
            </div>
            {#if !expandedRows.has(index) && hoveredCell?.row === index && hoveredCell?.col === column}
              <button
                class="absolute bottom-0.5 right-0.5 text-xs px-1 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700"
                onclick={(e) => {
                  e.stopPropagation();
                  toggleRowExpansion(index);
                }}
              >
                ↘
              </button>
            {/if}
          </td>
        {/each}
        <td></td>
      </tr>
    {/each}
  </tbody>
</table>
