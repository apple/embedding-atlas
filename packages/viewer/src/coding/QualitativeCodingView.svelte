<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import * as SQL from "@uwdata/mosaic-sql";

  import type { ChartContext, RowID } from "../charts/chart.js";
  import type { ColumnStyle } from "../renderers/index.js";

  import { IconSidebar, IconTag } from "../assets/icons.js";
  import Button from "../widgets/Button.svelte";

  import CodingPanel from "./CodingPanel.svelte";
  import EmbeddingDetailsModal from "./EmbeddingDetailsModal.svelte";
  import SelectionToolbar from "./SelectionToolbar.svelte";
  import { createCodingStore, type CodingStore } from "./store.svelte.js";
  import { createSelectionStore, type SelectionStore } from "./selection.svelte.js";
  import { createEditStore, type EditStore } from "./editable.js";
  import type { Code, CodingState } from "./types.js";

  interface Props {
    context: ChartContext;
    columnStyles?: Record<string, ColumnStyle>;
    initialCodingState?: CodingState | null;
    onCodingStateChange?: (state: CodingState) => void;
    children?: import("svelte").Snippet;
  }

  let {
    context,
    columnStyles = {},
    initialCodingState = null,
    onCodingStateChange,
    children,
  }: Props = $props();

  // Create stores
  const codingStore: CodingStore = createCodingStore(initialCodingState ?? undefined);
  const selectionStore: SelectionStore = createSelectionStore();
  const editStore: EditStore = createEditStore();

  // UI state
  let showCodingPanel = $state(true);
  let showDetailsModal = $state(false);
  let detailsDataPointId = $state<RowID | null>(null);
  let detailsData = $state<Record<string, any> | null>(null);

  // Track coding state changes
  $effect(() => {
    const state = codingStore.exportState();
    onCodingStateChange?.(state);
  });

  // Visible columns for the modal
  let visibleColumns = $derived(
    context.columns
      .filter((c) => !c.name.startsWith("__"))
      .map((c) => c.name)
  );

  // Query data for the details modal
  async function queryDataPoint(id: RowID): Promise<Record<string, any> | null> {
    try {
      const result = await context.coordinator.query(
        SQL.Query.from(context.table)
          .select(Object.fromEntries(visibleColumns.map((c) => [c, SQL.column(c)])))
          .where(SQL.eq(SQL.column(context.id), SQL.literal(id)))
      );
      if (result.numRows > 0) {
        return Object.fromEntries(
          visibleColumns.map((col) => [col, result.get(0)[col]])
        );
      }
      return null;
    } catch (e) {
      console.error("Failed to query data point:", e);
      return null;
    }
  }

  // Open details modal for a data point
  async function openDetails(id: RowID) {
    detailsDataPointId = id;
    detailsData = await queryDataPoint(id);
    showDetailsModal = true;
  }

  // Handle selection from embedding view
  export function handleSelection(points: { identifier: RowID }[] | null) {
    if (points && points.length > 0) {
      selectionStore.selectMultiple(points.map((p) => p.identifier));
    }
  }

  // Handle click on a point
  export function handlePointClick(id: RowID, event?: MouseEvent) {
    if (event) {
      selectionStore.handleClick(id, {
        shiftKey: event.shiftKey,
        ctrlKey: event.ctrlKey,
        metaKey: event.metaKey,
      });
    } else {
      selectionStore.select(id);
    }
  }

  // Handle double-click to open details
  export function handlePointDoubleClick(id: RowID) {
    openDetails(id);
  }

  // Navigation in details modal
  let allDataPoints = $state<RowID[]>([]);
  let currentIndex = $derived(
    detailsDataPointId !== null ? allDataPoints.indexOf(detailsDataPointId) : -1
  );

  async function loadAllDataPoints() {
    try {
      const result = await context.coordinator.query(
        SQL.Query.from(context.table)
          .select({ id: SQL.column(context.id) })
          .orderby(context.id)
      );
      allDataPoints = Array.from(result).map((row: any) => row.id);
    } catch (e) {
      console.error("Failed to load data points:", e);
    }
  }

  async function navigateDetails(direction: "prev" | "next") {
    if (allDataPoints.length === 0) {
      await loadAllDataPoints();
    }

    if (currentIndex === -1) return;

    let newIndex: number;
    if (direction === "prev") {
      newIndex = currentIndex > 0 ? currentIndex - 1 : allDataPoints.length - 1;
    } else {
      newIndex = currentIndex < allDataPoints.length - 1 ? currentIndex + 1 : 0;
    }

    const newId = allDataPoints[newIndex];
    detailsDataPointId = newId;
    detailsData = await queryDataPoint(newId);
  }

  // Export/import coding state
  function handleExportCodes() {
    const state = codingStore.exportState();
    const blob = new Blob([JSON.stringify(state, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "qualitative-codes.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleImportCodes() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      try {
        const text = await file.text();
        const state = JSON.parse(text) as CodingState;
        codingStore.importState(state);
      } catch (err) {
        console.error("Failed to import codes:", err);
        alert("Failed to import codes. Please check the file format.");
      }
    };
    input.click();
  }

  // Expose stores and methods for parent components
  export { codingStore, selectionStore, editStore };
</script>

<div class="qualitative-coding-view flex h-full">
  <!-- Main content area -->
  <div class="flex-1 min-w-0 flex flex-col relative">
    <!-- Coding mode toggle button -->
    <div class="absolute top-2 right-2 z-10 flex gap-2">
      <Button
        icon={IconTag}
        title={showCodingPanel ? "Hide coding panel" : "Show coding panel"}
        onClick={() => (showCodingPanel = !showCodingPanel)}
      />
    </div>

    <!-- Child content (embedding view, table, etc.) -->
    <div class="flex-1 overflow-hidden">
      {@render children?.()}
    </div>

    <!-- Selection toolbar -->
    <SelectionToolbar
      {selectionStore}
      {codingStore}
      onViewDetails={() => {
        const selected = selectionStore.selectedArray;
        if (selected.length === 1) {
          openDetails(selected[0]);
        }
      }}
    />
  </div>

  <!-- Coding panel sidebar -->
  {#if showCodingPanel}
    <div class="w-80 flex-shrink-0 border-l border-slate-200 dark:border-slate-700">
      <CodingPanel
        {codingStore}
        onExport={handleExportCodes}
        onImport={handleImportCodes}
        onCodeClick={(code: Code) => {
          // TODO: Filter embedding view to show only data points with this code
          console.log("Code clicked:", code);
        }}
      />
    </div>
  {/if}

  <!-- Details modal -->
  <EmbeddingDetailsModal
    open={showDetailsModal}
    dataPointId={detailsDataPointId}
    data={detailsData}
    {context}
    {codingStore}
    columnStyles={columnStyles}
    visibleColumns={visibleColumns}
    onClose={() => (showDetailsModal = false)}
    onNavigate={navigateDetails}
  />
</div>

<style>
  .qualitative-coding-view {
    width: 100%;
    height: 100%;
  }
</style>
