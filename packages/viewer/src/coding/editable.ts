// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { CustomCell } from "@embedding-atlas/table";
import { createClassComponent } from "svelte/legacy";

import EditableCell from "./EditableCell.svelte";

export interface CellEdit {
  rowId: any;
  column: string;
  oldValue: any;
  newValue: any;
  timestamp: number;
}

/** Store for tracking cell edits */
export function createEditStore() {
  let edits = $state<CellEdit[]>([]);
  let editsByCell = $state<Map<string, any>>(new Map());

  function getCellKey(rowId: any, column: string): string {
    return `${rowId}:${column}`;
  }

  return {
    get edits() {
      return edits;
    },

    get editsByCell() {
      return editsByCell;
    },

    /** Record a cell edit */
    recordEdit(rowId: any, column: string, oldValue: any, newValue: any) {
      const edit: CellEdit = {
        rowId,
        column,
        oldValue,
        newValue,
        timestamp: Date.now(),
      };
      edits = [...edits, edit];

      const key = getCellKey(rowId, column);
      const newMap = new Map(editsByCell);
      newMap.set(key, newValue);
      editsByCell = newMap;
    },

    /** Get the current value for a cell (returns edited value if exists) */
    getValue(rowId: any, column: string, originalValue: any): any {
      const key = getCellKey(rowId, column);
      if (editsByCell.has(key)) {
        return editsByCell.get(key);
      }
      return originalValue;
    },

    /** Check if a cell has been edited */
    isEdited(rowId: any, column: string): boolean {
      const key = getCellKey(rowId, column);
      return editsByCell.has(key);
    },

    /** Undo the last edit */
    undo(): CellEdit | null {
      if (edits.length === 0) return null;

      const lastEdit = edits[edits.length - 1];
      edits = edits.slice(0, -1);

      const key = getCellKey(lastEdit.rowId, lastEdit.column);
      const newMap = new Map(editsByCell);

      // Find if there's a previous edit for this cell
      const previousEdit = [...edits].reverse().find(
        (e) => e.rowId === lastEdit.rowId && e.column === lastEdit.column
      );

      if (previousEdit) {
        newMap.set(key, previousEdit.newValue);
      } else {
        newMap.delete(key);
      }

      editsByCell = newMap;
      return lastEdit;
    },

    /** Clear all edits */
    clear() {
      edits = [];
      editsByCell = new Map();
    },

    /** Get all edits for a specific column */
    getEditsForColumn(column: string): CellEdit[] {
      return edits.filter((e) => e.column === column);
    },

    /** Get all edits for a specific row */
    getEditsForRow(rowId: any): CellEdit[] {
      return edits.filter((e) => e.rowId === rowId);
    },

    /** Export edits as a map of rowId -> { column: value } */
    exportEdits(): Record<string, Record<string, any>> {
      const result: Record<string, Record<string, any>> = {};
      for (const [key, value] of editsByCell) {
        const [rowId, column] = key.split(":");
        if (!result[rowId]) {
          result[rowId] = {};
        }
        result[rowId][column] = value;
      }
      return result;
    },

    /** Import edits from a map */
    importEdits(data: Record<string, Record<string, any>>) {
      const newMap = new Map<string, any>();
      for (const [rowId, columns] of Object.entries(data)) {
        for (const [column, value] of Object.entries(columns)) {
          const key = getCellKey(rowId, column);
          newMap.set(key, value);
        }
      }
      editsByCell = newMap;
    },
  };
}

export type EditStore = ReturnType<typeof createEditStore>;

/** Create an editable custom cell for the table */
export function createEditableCell(
  column: string,
  rowKey: string,
  onSave: (rowId: any, column: string, newValue: any) => void
): CustomCell {
  return class EditableCellWrapper {
    private component: any;

    constructor(target: HTMLElement, props: { value: any; rowData: Record<string, any> }) {
      this.component = createClassComponent({
        component: EditableCell,
        target,
        props: {
          value: props.value,
          rowData: props.rowData,
          rowKey,
          column,
          onSave,
        },
      });
    }

    update(props: { value: any; rowData: Record<string, any> }) {
      this.component.$set({
        value: props.value,
        rowData: props.rowData,
      });
    }

    destroy() {
      this.component.$destroy();
    }
  };
}

/** Create editable cells for multiple columns */
export function createEditableCells(
  columns: string[],
  rowKey: string,
  onSave: (rowId: any, column: string, newValue: any) => void
): Record<string, CustomCell> {
  const cells: Record<string, CustomCell> = {};
  for (const column of columns) {
    cells[column] = createEditableCell(column, rowKey, onSave);
  }
  return cells;
}
