<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onMount } from "svelte";

  import HorizontalScrollbar from "./views/HorizontalScrollbar.svelte";
  import VerticalScrollbar from "./views/VerticalScrollbar.svelte";
  import Cell from "./views/cells/Cell.svelte";
  import HeaderRow from "./views/headers/HeaderRow.svelte";

  import type { TableProps } from ".";
  import StyleWrapper from "./StyleWrapper.svelte";
  import { rowNumber } from "./api/config";
  import { ConfigContext } from "./context/config.svelte";
  import { Context } from "./context/context.svelte";
  import { CoordinatorContext } from "./context/coordinator.svelte";
  import { CustomCellsContext } from "./context/custom-cells.svelte";
  import { CustomHeadersContext } from "./context/custom-headers.svelte";
  import { StyleContext } from "./context/style.svelte";
  import { OID } from "./mosaic-clients/RowsClient.js";
  import { add, diff, remove } from "./util.js";
  import RowBackground from "./views/RowBackground.svelte";

  let {
    table: tableName,
    columns,
    rowKey,
    columnConfigs,
    onColumnConfigsChange,
    showRowNumber,
    onShowRowNumberChange,
    coordinator,
    filter,
    scrollTo,
    colorScheme,
    theme,
    lineHeight,
    numLines,
    customCells,
    additionalHeaderContents,
    headerHeight,
    onRowClick,
    highlightedRows,
    highlightHoveredRow,
  }: TableProps = $props();

  ConfigContext.initialize();
  CustomCellsContext.initialize();
  StyleContext.initialize();
  Context.initialize();
  const controller = Context.controller;
  const model = Context.model;
  const overscrollModifier = Context.overscrollModifier;
  const config = ConfigContext.config;
  const style = StyleContext.style;

  $effect(() => {
    if (scrollTo != null) {
      controller.scrollToRow(String(scrollTo));
    }
  });

  $effect(() => {
    if (highlightedRows && highlightedRows.length > 0) {
      config.highlightedRows = new Set(highlightedRows.map((r) => String(r)));
    } else {
      config.highlightedRows = null;
    }
  });

  $effect(() => {
    if (onRowClick != null) {
      config.onRowClick = onRowClick;
    } else {
      config.onRowClick = null;
    }
  });

  $effect(() => {
    if (coordinator) {
      CoordinatorContext.coordinator = coordinator;
    } else {
      CoordinatorContext.coordinator = null;
    }
  });

  $effect(() => {
    if (numLines != null) {
      config.textMaxLines = numLines;
    } else {
      config.textMaxLines = config.DEFAULT_TEXT_MAX_LINES;
    }

    if (lineHeight != null) {
      config.lineHeight = lineHeight;
    } else {
      config.lineHeight = config.DEFAULT_LINE_HEIGHT;
    }
  });

  $effect(() => {
    if (colorScheme != null) {
      style.colorScheme = colorScheme;
    } else {
      style.colorScheme = null;
    }
  });

  $effect(() => {
    // update styles
    if (theme != null) {
      style.theme = theme;
    } else {
      style.theme = {};
    }

    if (colorScheme != null) {
      style.colorScheme = colorScheme;
    } else {
      style.colorScheme = null;
    }
  });

  $effect(() => {
    if (columnConfigs != null) {
      config.columnConfigs = columnConfigs;
    } else {
      config.columnConfigs = {};
    }

    if (onColumnConfigsChange != null) {
      config.onColumnConfigsChange = onColumnConfigsChange;
    } else {
      config.onColumnConfigsChange = () => {};
    }
  });

  $effect(() => {
    config.showRowNumber = showRowNumber ?? null;
  });

  $effect(() => {
    config.onShowRowNumberChange = onShowRowNumberChange ?? null;
  });

  $effect(() => {
    // update table
    controller.initialize({
      tableName,
      rowKey,
      columns: [rowNumber(), ...columns],
      filterBy: filter ?? null,
    });
  });

  $effect(() => {
    if (customCells != null) {
      CustomCellsContext.config = customCells;
    } else {
      CustomCellsContext.config = {};
    }
  });

  $effect(() => {
    if (additionalHeaderContents != null) {
      CustomHeadersContext.config = additionalHeaderContents;
    } else {
      CustomHeadersContext.config = {};
    }
  });

  $effect(() => {
    if (headerHeight != null) {
      config.headerHeight = headerHeight;
    } else {
      config.headerHeight = null;
    }
  });

  $effect(() => {
    if (highlightHoveredRow != null) {
      config.highlightHoveredRow = highlightHoveredRow;
    } else {
      config.highlightHoveredRow = false;
    }
  });

  let rows: string[] = $state.raw([]);
  let updateKey = $state(0);
  let scrollContainer: HTMLElement | null = $state(null);
  let tickVisibleRows: string[] = $state([]);
  let tickVisibleCols: string[] = $state([]);
  let renderVisibleRows: string[] = $derived.by(() => {
    // the render visible rows might be out of sync if a filter gets applied between ticks
    // so we need to check to make sure the data still exists and hasn't been deleted by the filter.
    return tickVisibleRows.filter((r) => controller.rowStillExists(r));
  });
  let renderVisibleCols: string[] = $derived(tickVisibleCols);

  let lastRAFId = 0;

  onMount(() => {
    lastRAFId = requestAnimationFrame(tick);
    return () => {
      controller.teardown();
      model.teardown();
      cancelAnimationFrame(lastRAFId);
    };
  });

  function distanceFromExistingRows(row: string, existing: string[]): number {
    if (existing.length > 0) {
      const lastRow = existing[existing.length - 1];
      return Math.abs(model.data[row][OID] - model.data[lastRow][OID]);
    }

    return 0;
  }

  function resolveRows() {
    const { left, right } = diff(rows, model.renderableRows);
    if (left.length === 0 && right.length === 0) {
      return;
    }
    rows = remove(rows, left); // remove the rows that have been deleted from the model
    rows = add(
      rows, // add the rows that have been added by the model
      right
        .sort((a, b) => distanceFromExistingRows(a, rows) - distanceFromExistingRows(b, rows))
        .slice(0, controller.isJumping ? controller.rowsOnScreen : config.rowRenderBatchSize),
    );
  }

  function tick() {
    resolveRows();

    tickVisibleRows = rows.filter((row) => controller.rowIsVisible(row));
    tickVisibleCols = model.renderableCols.filter((col) => controller.colIsVisible(col));

    const xScroll = controller.xScroll;
    const yScroll = overscrollModifier.yScroll(controller.yScroll);
    if (scrollContainer) {
      scrollContainer.style.transform = `translate3d(${xScroll}px, ${yScroll}px, 0)`;
    }
    updateKey = controller.updateKey;

    lastRAFId = requestAnimationFrame(tick);
  }
</script>

<StyleWrapper>
  <div class="table" bind:this={controller.element} onwheel={controller.handleWheel}>
    <HeaderRow />
    <div class="table-contents" bind:clientHeight={controller.viewHeight} bind:clientWidth={controller.viewWidth}>
      {#if controller.isReady}
        <div class="scroll-container" bind:this={scrollContainer}>
          <!-- use key to refresh if rows get replaced-->
          {#key updateKey}
            {#each renderVisibleRows as row (row)}
              {#each renderVisibleCols as col (col)}
                <Cell row={row} col={col} />
              {/each}
            {/each}
            {#each model.renderableRows as row (row)}
              <RowBackground row={row} />
            {/each}
          {/key}
        </div>
        <VerticalScrollbar />
        <HorizontalScrollbar />
      {/if}
    </div>
  </div>
</StyleWrapper>

<style>
  .table {
    width: 100%;
    max-width: var(--max-width);
    height: 100%;
    display: flex;
    flex-direction: column;
    position: relative;
  }

  .table-contents {
    position: relative;
    overflow: hidden;
    flex-grow: 1;
  }

  .scroll-container {
    position: absolute;
    width: 0;
    height: 0;
    will-change: transform;
    contain: layout size;
  }
</style>
