<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { CustomCell } from "@embedding-atlas/table";
  import { Table } from "@embedding-atlas/table/svelte";

  import { getRenderer, type ColumnStyle } from "../../renderers/index.js";
  import type { ColumnDesc } from "../../utils/database.js";
  import { isolatedWritable } from "../../utils/store.js";
  import { type ChartViewProps, type RowID } from "../chart.js";
  import { tableTheme } from "./table_theme.js";
  import type { TableSpec } from "./types.js";

  interface State {}

  let { context, spec }: ChartViewProps<TableSpec, State> = $props();

  let { colorScheme, columnStyles } = context;
  let highlightStore = isolatedWritable(context.highlight);

  let scrollTo = $state<RowID | null>(null);

  $effect.pre(() =>
    highlightStore.subscribe((id) => {
      if (scrollTo !== id) {
        scrollTo = id;
      }
    }),
  );

  function resolveCustomCellRenderers(
    columns: ColumnDesc[],
    columnStyles: Record<string, ColumnStyle>,
    tableCellRenderers: Record<string, string | CustomCell> | null | undefined,
  ) {
    let result: Record<string, any> = {};
    for (let column of columns) {
      if (tableCellRenderers?.[column.name] != null) {
        result[column.name] = getRenderer(tableCellRenderers[column.name]);
      }
      if (columnStyles[column.name]?.renderer != null) {
        result[column.name] = getRenderer(columnStyles[column.name]?.renderer);
      }
    }
    return result;
  }

  let resolvedCustomCellRenderers = $derived(
    resolveCustomCellRenderers(context.columns, $columnStyles, context.tableCellRenderers),
  );

  let resolvedTableTheme = $derived(
    context.tableTheme ? { ...tableTheme, ...context.tableTheme } : tableTheme,
  );
</script>

<Table
  coordinator={context.coordinator}
  table={context.table}
  rowKey={context.id}
  columns={spec.columns}
  filter={context.filter}
  scrollTo={scrollTo}
  onRowClick={async (identifier) => {
    highlightStore.set(identifier);
  }}
  numLines={3}
  colorScheme={$colorScheme}
  theme={resolvedTableTheme}
  highlightHoveredRow={true}
  customCells={resolvedCustomCellRenderers}
/>
