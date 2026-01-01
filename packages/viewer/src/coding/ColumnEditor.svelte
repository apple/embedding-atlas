<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconPlus, IconClose, IconEdit } from "../assets/icons.js";
  import Modal from "../widgets/Modal.svelte";

  interface Props {
    columns: string[];
    onAddColumn: (name: string, defaultValue?: string) => void;
    onRenameColumn: (oldName: string, newName: string) => void;
    onRemoveColumn: (name: string) => void;
    protectedColumns?: string[];
  }

  let {
    columns,
    onAddColumn,
    onRenameColumn,
    onRemoveColumn,
    protectedColumns = ["__row_index__", "id"],
  }: Props = $props();

  let showAddModal = $state(false);
  let showRenameModal = $state(false);
  let newColumnName = $state("");
  let newColumnDefault = $state("");
  let renameOldName = $state("");
  let renameNewName = $state("");
  let error = $state<string | null>(null);

  function isProtected(column: string): boolean {
    return protectedColumns.includes(column);
  }

  function handleAddColumn() {
    const name = newColumnName.trim();
    if (!name) {
      error = "Column name is required";
      return;
    }
    if (columns.includes(name)) {
      error = "Column already exists";
      return;
    }
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
      error = "Invalid column name (use letters, numbers, underscores)";
      return;
    }

    onAddColumn(name, newColumnDefault.trim() || undefined);
    showAddModal = false;
    newColumnName = "";
    newColumnDefault = "";
    error = null;
  }

  function openRenameModal(column: string) {
    renameOldName = column;
    renameNewName = column;
    error = null;
    showRenameModal = true;
  }

  function handleRenameColumn() {
    const newName = renameNewName.trim();
    if (!newName) {
      error = "Column name is required";
      return;
    }
    if (newName !== renameOldName && columns.includes(newName)) {
      error = "Column already exists";
      return;
    }
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(newName)) {
      error = "Invalid column name (use letters, numbers, underscores)";
      return;
    }

    if (newName !== renameOldName) {
      onRenameColumn(renameOldName, newName);
    }
    showRenameModal = false;
    renameOldName = "";
    renameNewName = "";
    error = null;
  }

  function handleRemoveColumn(column: string) {
    if (confirm(`Are you sure you want to remove column "${column}"? This cannot be undone.`)) {
      onRemoveColumn(column);
    }
  }
</script>

<div class="column-editor">
  <div class="header">
    <h4 class="text-sm font-semibold text-slate-700 dark:text-slate-300">
      Columns ({columns.length})
    </h4>
    <button
      onclick={() => {
        showAddModal = true;
        error = null;
      }}
      class="add-button"
      title="Add new column"
    >
      <IconPlus class="w-4 h-4" />
      Add Column
    </button>
  </div>

  <div class="column-list">
    {#each columns as column}
      <div class="column-item" class:protected={isProtected(column)}>
        <span class="column-name" title={column}>
          {column}
        </span>
        {#if !isProtected(column)}
          <div class="column-actions">
            <button
              onclick={() => openRenameModal(column)}
              class="action-button"
              title="Rename column"
            >
              <IconEdit class="w-3 h-3" />
            </button>
            <button
              onclick={() => handleRemoveColumn(column)}
              class="action-button danger"
              title="Remove column"
            >
              <IconClose class="w-3 h-3" />
            </button>
          </div>
        {:else}
          <span class="protected-badge">system</span>
        {/if}
      </div>
    {/each}
  </div>
</div>

<!-- Add Column Modal -->
<Modal
  open={showAddModal}
  size="sm"
  title="Add New Column"
  onClose={() => {
    showAddModal = false;
    error = null;
  }}
>
  <div class="modal-content">
    <div class="form-group">
      <label for="column-name">Column Name</label>
      <input
        id="column-name"
        type="text"
        bind:value={newColumnName}
        placeholder="e.g., category, notes"
        class="form-input"
        onkeydown={(e) => e.key === "Enter" && handleAddColumn()}
      />
    </div>

    <div class="form-group">
      <label for="default-value">Default Value (optional)</label>
      <input
        id="default-value"
        type="text"
        bind:value={newColumnDefault}
        placeholder="Leave empty for blank"
        class="form-input"
      />
    </div>

    {#if error}
      <div class="error-message">{error}</div>
    {/if}
  </div>

  {#snippet footer()}
    <div class="modal-footer">
      <button
        onclick={() => {
          showAddModal = false;
          error = null;
        }}
        class="btn-secondary"
      >
        Cancel
      </button>
      <button
        onclick={handleAddColumn}
        disabled={!newColumnName.trim()}
        class="btn-primary"
      >
        Add Column
      </button>
    </div>
  {/snippet}
</Modal>

<!-- Rename Column Modal -->
<Modal
  open={showRenameModal}
  size="sm"
  title="Rename Column"
  onClose={() => {
    showRenameModal = false;
    error = null;
  }}
>
  <div class="modal-content">
    <div class="form-group">
      <label for="rename-column">New Name for "{renameOldName}"</label>
      <input
        id="rename-column"
        type="text"
        bind:value={renameNewName}
        class="form-input"
        onkeydown={(e) => e.key === "Enter" && handleRenameColumn()}
      />
    </div>

    {#if error}
      <div class="error-message">{error}</div>
    {/if}
  </div>

  {#snippet footer()}
    <div class="modal-footer">
      <button
        onclick={() => {
          showRenameModal = false;
          error = null;
        }}
        class="btn-secondary"
      >
        Cancel
      </button>
      <button
        onclick={handleRenameColumn}
        disabled={!renameNewName.trim()}
        class="btn-primary"
      >
        Rename
      </button>
    </div>
  {/snippet}
</Modal>

<style>
  .column-editor {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .add-button {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    color: #3b82f6;
    background: transparent;
    border: 1px solid #3b82f6;
    border-radius: 0.25rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .add-button:hover {
    background: #3b82f6;
    color: white;
  }

  .column-list {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    max-height: 300px;
    overflow-y: auto;
  }

  .column-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.375rem 0.5rem;
    background: #f8fafc;
    border-radius: 0.25rem;
    font-size: 0.875rem;
  }

  :global(.dark) .column-item {
    background: #334155;
  }

  .column-item.protected {
    opacity: 0.6;
  }

  .column-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: #475569;
  }

  :global(.dark) .column-name {
    color: #cbd5e1;
  }

  .column-actions {
    display: flex;
    gap: 0.25rem;
  }

  .action-button {
    padding: 0.25rem;
    color: #64748b;
    background: transparent;
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .action-button:hover {
    background: #e2e8f0;
    color: #334155;
  }

  :global(.dark) .action-button:hover {
    background: #475569;
    color: #e2e8f0;
  }

  .action-button.danger:hover {
    background: #fef2f2;
    color: #dc2626;
  }

  :global(.dark) .action-button.danger:hover {
    background: #450a0a;
    color: #fca5a5;
  }

  .protected-badge {
    font-size: 0.625rem;
    padding: 0.125rem 0.375rem;
    color: #64748b;
    background: #e2e8f0;
    border-radius: 9999px;
  }

  :global(.dark) .protected-badge {
    background: #475569;
    color: #94a3b8;
  }

  .modal-content {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .form-group label {
    font-size: 0.875rem;
    font-weight: 500;
    color: #475569;
  }

  :global(.dark) .form-group label {
    color: #cbd5e1;
  }

  .form-input {
    padding: 0.5rem 0.75rem;
    font-size: 0.875rem;
    border: 1px solid #cbd5e1;
    border-radius: 0.375rem;
    background: white;
    color: #1e293b;
  }

  :global(.dark) .form-input {
    border-color: #475569;
    background: #1e293b;
    color: #e2e8f0;
  }

  .form-input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .error-message {
    padding: 0.5rem;
    font-size: 0.875rem;
    color: #dc2626;
    background: #fef2f2;
    border-radius: 0.25rem;
  }

  :global(.dark) .error-message {
    background: #450a0a;
    color: #fca5a5;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
  }

  .btn-primary,
  .btn-secondary {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    font-weight: 500;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-primary {
    background: #3b82f6;
    color: white;
    border: none;
  }

  .btn-primary:hover:not(:disabled) {
    background: #2563eb;
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: transparent;
    color: #64748b;
    border: 1px solid #cbd5e1;
  }

  :global(.dark) .btn-secondary {
    border-color: #475569;
    color: #94a3b8;
  }

  .btn-secondary:hover {
    background: #f1f5f9;
  }

  :global(.dark) .btn-secondary:hover {
    background: #334155;
  }
</style>
