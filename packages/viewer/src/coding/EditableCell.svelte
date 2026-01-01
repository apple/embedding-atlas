<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconCheck, IconClose, IconEdit } from "../assets/icons.js";

  interface Props {
    value: any;
    rowData: Record<string, any>;
    rowKey?: string;
    column?: string;
    onSave?: (rowId: any, column: string, newValue: any) => void;
  }

  let { value, rowData, rowKey = "__row_index__", column = "", onSave }: Props = $props();

  let isEditing = $state(false);
  let editValue = $state("");

  function startEdit() {
    editValue = value?.toString() ?? "";
    isEditing = true;
  }

  function cancelEdit() {
    isEditing = false;
    editValue = "";
  }

  function saveEdit() {
    if (onSave && column) {
      const rowId = rowData[rowKey];
      onSave(rowId, column, editValue);
    }
    isEditing = false;
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      saveEdit();
    } else if (e.key === "Escape") {
      cancelEdit();
    }
  }

  function formatValue(val: any): string {
    if (val === null || val === undefined) return "(empty)";
    if (typeof val === "object") return JSON.stringify(val);
    return String(val);
  }
</script>

<div class="editable-cell">
  {#if isEditing}
    <div class="edit-container">
      <textarea
        bind:value={editValue}
        onkeydown={handleKeydown}
        class="edit-input"
        rows="2"
      ></textarea>
      <div class="edit-actions">
        <button onclick={saveEdit} class="save-btn" title="Save">
          <IconCheck class="w-4 h-4" />
        </button>
        <button onclick={cancelEdit} class="cancel-btn" title="Cancel">
          <IconClose class="w-4 h-4" />
        </button>
      </div>
    </div>
  {:else}
    <div class="view-container">
      <span class="cell-value">{formatValue(value)}</span>
      <button onclick={startEdit} class="edit-btn" title="Edit">
        <IconEdit class="w-4 h-4" />
      </button>
    </div>
  {/if}
</div>

<style>
  .editable-cell {
    width: 100%;
    min-height: 1.5em;
  }

  .view-container {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
  }

  .cell-value {
    flex: 1;
    word-break: break-word;
  }

  .edit-btn {
    flex-shrink: 0;
    opacity: 0;
    padding: 0.25rem;
    border-radius: 0.25rem;
    color: var(--secondary-text-color, #64748b);
    background: transparent;
    border: none;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .view-container:hover .edit-btn {
    opacity: 1;
  }

  .edit-btn:hover {
    background: var(--hover-bg, rgba(0, 0, 0, 0.05));
  }

  .edit-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .edit-input {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid var(--outline-color, #cbd5e1);
    border-radius: 0.25rem;
    font-family: inherit;
    font-size: inherit;
    resize: vertical;
    min-height: 2.5em;
  }

  .edit-input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .edit-actions {
    display: flex;
    gap: 0.25rem;
    justify-content: flex-end;
  }

  .save-btn,
  .cancel-btn {
    padding: 0.25rem 0.5rem;
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.75rem;
  }

  .save-btn {
    background: #22c55e;
    color: white;
  }

  .save-btn:hover {
    background: #16a34a;
  }

  .cancel-btn {
    background: #e2e8f0;
    color: #475569;
  }

  .cancel-btn:hover {
    background: #cbd5e1;
  }
</style>
