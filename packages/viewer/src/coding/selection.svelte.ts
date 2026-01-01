// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { RowID } from "../charts/chart.js";

/** Selection mode for multi-selection */
export type SelectionMode = "single" | "multi" | "range";

/** Create a reactive selection store for managing selected data points */
export function createSelectionStore() {
  let selectedIds = $state<Set<RowID>>(new Set());
  let lastSelectedId = $state<RowID | null>(null);
  let selectionMode = $state<SelectionMode>("single");
  let isSelecting = $state(false);

  return {
    get selectedIds() {
      return selectedIds;
    },

    get selectedArray(): RowID[] {
      return Array.from(selectedIds);
    },

    get count(): number {
      return selectedIds.size;
    },

    get lastSelectedId() {
      return lastSelectedId;
    },

    get selectionMode() {
      return selectionMode;
    },

    get isSelecting() {
      return isSelecting;
    },

    /** Check if an ID is selected */
    isSelected(id: RowID): boolean {
      return selectedIds.has(id);
    },

    /** Select a single item (replaces selection) */
    select(id: RowID) {
      selectedIds = new Set([id]);
      lastSelectedId = id;
    },

    /** Add an item to the selection */
    addToSelection(id: RowID) {
      const newSet = new Set(selectedIds);
      newSet.add(id);
      selectedIds = newSet;
      lastSelectedId = id;
    },

    /** Remove an item from the selection */
    removeFromSelection(id: RowID) {
      const newSet = new Set(selectedIds);
      newSet.delete(id);
      selectedIds = newSet;
      if (lastSelectedId === id) {
        lastSelectedId = newSet.size > 0 ? Array.from(newSet)[newSet.size - 1] : null;
      }
    },

    /** Toggle selection of an item */
    toggle(id: RowID) {
      if (selectedIds.has(id)) {
        this.removeFromSelection(id);
      } else {
        this.addToSelection(id);
      }
    },

    /** Select multiple items */
    selectMultiple(ids: RowID[]) {
      selectedIds = new Set(ids);
      lastSelectedId = ids.length > 0 ? ids[ids.length - 1] : null;
    },

    /** Add multiple items to selection */
    addMultipleToSelection(ids: RowID[]) {
      const newSet = new Set(selectedIds);
      for (const id of ids) {
        newSet.add(id);
      }
      selectedIds = newSet;
      if (ids.length > 0) {
        lastSelectedId = ids[ids.length - 1];
      }
    },

    /** Clear the selection */
    clear() {
      selectedIds = new Set();
      lastSelectedId = null;
    },

    /** Set the selection mode */
    setMode(mode: SelectionMode) {
      selectionMode = mode;
    },

    /** Handle click with modifier keys */
    handleClick(id: RowID, event: { shiftKey: boolean; ctrlKey: boolean; metaKey: boolean }) {
      const isModifierClick = event.ctrlKey || event.metaKey;

      if (selectionMode === "multi" || isModifierClick) {
        // Multi-select mode: toggle selection
        this.toggle(id);
      } else if (event.shiftKey && lastSelectedId !== null) {
        // Range selection would need row indices - for now just add to selection
        this.addToSelection(id);
      } else {
        // Single select mode: replace selection
        this.select(id);
      }
    },

    /** Start a selection operation (e.g., lasso) */
    startSelecting() {
      isSelecting = true;
    },

    /** End a selection operation */
    endSelecting() {
      isSelecting = false;
    },

    /** Set selecting state with IDs */
    setSelecting(ids: RowID[]) {
      selectedIds = new Set(ids);
      isSelecting = true;
    },

    /** Finalize selection after lasso/brush */
    finalizeSelection(ids: RowID[], mode: "replace" | "add" = "replace") {
      if (mode === "add") {
        this.addMultipleToSelection(ids);
      } else {
        this.selectMultiple(ids);
      }
      isSelecting = false;
    },
  };
}

export type SelectionStore = ReturnType<typeof createSelectionStore>;
