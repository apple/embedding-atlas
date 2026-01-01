<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconPlus, IconMemo, IconSearch, IconClose } from "../assets/icons.js";
  import type { Memo, Code } from "./types.js";
  import Modal from "../widgets/Modal.svelte";
  import MemoEditor from "./MemoEditor.svelte";
  import type { RowID } from "../charts/chart.js";
  import { generateId } from "./types.js";

  interface Props {
    memos: Record<string, Memo>;
    codes: Record<string, Code>;
    onCreateMemo: (memo: Memo) => void;
    onUpdateMemo: (memo: Memo) => void;
    onDeleteMemo: (memoId: string) => void;
    linkedDataPoints?: RowID[];
  }

  let {
    memos,
    codes,
    onCreateMemo,
    onUpdateMemo,
    onDeleteMemo,
    linkedDataPoints = [],
  }: Props = $props();

  let searchQuery = $state("");
  let filterType = $state<Memo["memoType"] | "all">("all");
  let filterCode = $state<string | null>(null);
  let showEditor = $state(false);
  let editingMemo = $state<Memo | null>(null);

  const memoTypes: { value: Memo["memoType"] | "all"; label: string }[] = [
    { value: "all", label: "All" },
    { value: "theoretical", label: "Theoretical" },
    { value: "methodological", label: "Methodological" },
    { value: "observational", label: "Observational" },
  ];

  let filteredMemos = $derived.by(() => {
    let result = Object.values(memos);

    // Filter by type
    if (filterType !== "all") {
      result = result.filter((m) => m.memoType === filterType);
    }

    // Filter by linked code
    if (filterCode) {
      result = result.filter((m) => m.linkedCodes.includes(filterCode!));
    }

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (m) =>
          m.content.toLowerCase().includes(query) ||
          m.tags.some((t) => t.includes(query))
      );
    }

    // Sort by creation date (newest first)
    return result.sort((a, b) => b.createdAt - a.createdAt);
  });

  function openNewMemo() {
    editingMemo = null;
    showEditor = true;
  }

  function openEditMemo(memo: Memo) {
    editingMemo = memo;
    showEditor = true;
  }

  function handleSave(memoData: Omit<Memo, "id" | "createdAt">) {
    if (editingMemo) {
      onUpdateMemo({
        ...editingMemo,
        ...memoData,
      });
    } else {
      onCreateMemo({
        id: generateId(),
        createdAt: Date.now(),
        ...memoData,
      });
    }
    showEditor = false;
    editingMemo = null;
  }

  function handleDelete() {
    if (editingMemo) {
      onDeleteMemo(editingMemo.id);
      showEditor = false;
      editingMemo = null;
    }
  }

  function formatDate(timestamp: number): string {
    return new Date(timestamp).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }

  function getMemoTypeColor(type: Memo["memoType"]): string {
    switch (type) {
      case "theoretical":
        return "#8b5cf6"; // violet
      case "methodological":
        return "#f59e0b"; // amber
      case "observational":
        return "#22c55e"; // green
      default:
        return "#64748b";
    }
  }

  let codeList = $derived(Object.values(codes));
</script>

<div class="memo-panel">
  <div class="panel-header">
    <div class="header-title">
      <IconMemo class="w-4 h-4 text-slate-500" />
      <h3 class="text-sm font-semibold text-slate-700 dark:text-slate-300">
        Memos ({Object.keys(memos).length})
      </h3>
    </div>
    <button onclick={openNewMemo} class="new-button" title="New memo">
      <IconPlus class="w-4 h-4" />
    </button>
  </div>

  <!-- Filters -->
  <div class="filters">
    <div class="search-box">
      <IconSearch class="w-4 h-4 text-slate-400" />
      <input
        type="text"
        bind:value={searchQuery}
        placeholder="Search memos..."
        class="search-input"
      />
      {#if searchQuery}
        <button onclick={() => (searchQuery = "")} class="clear-search">
          <IconClose class="w-3 h-3" />
        </button>
      {/if}
    </div>

    <div class="filter-row">
      <select
        bind:value={filterType}
        class="filter-select"
      >
        {#each memoTypes as type}
          <option value={type.value}>{type.label}</option>
        {/each}
      </select>

      {#if codeList.length > 0}
        <select
          bind:value={filterCode}
          class="filter-select"
        >
          <option value={null}>All codes</option>
          {#each codeList as code}
            <option value={code.id}>{code.name}</option>
          {/each}
        </select>
      {/if}
    </div>
  </div>

  <!-- Memo List -->
  <div class="memo-list">
    {#if filteredMemos.length === 0}
      <div class="empty-state">
        <IconMemo class="w-8 h-8 text-slate-300 dark:text-slate-600" />
        <p class="text-sm text-slate-400 dark:text-slate-500">
          {searchQuery || filterType !== "all" || filterCode
            ? "No memos match your filters"
            : "No memos yet. Create one to start!"}
        </p>
      </div>
    {:else}
      {#each filteredMemos as memo}
        <button
          class="memo-card"
          onclick={() => openEditMemo(memo)}
        >
          <div class="memo-header">
            <span
              class="memo-type-badge"
              style:background-color={getMemoTypeColor(memo.memoType)}
            >
              {memo.memoType}
            </span>
            <span class="memo-date">{formatDate(memo.createdAt)}</span>
          </div>

          <p class="memo-content">{memo.content}</p>

          {#if memo.linkedCodes.length > 0}
            <div class="memo-codes">
              {#each memo.linkedCodes.slice(0, 3) as codeId}
                {@const code = codes[codeId]}
                {#if code}
                  <span class="code-badge">
                    <span
                      class="code-dot"
                      style:background-color={code.color}
                    ></span>
                    {code.name}
                  </span>
                {/if}
              {/each}
              {#if memo.linkedCodes.length > 3}
                <span class="more-codes">+{memo.linkedCodes.length - 3}</span>
              {/if}
            </div>
          {/if}

          {#if memo.tags.length > 0}
            <div class="memo-tags">
              {#each memo.tags.slice(0, 3) as tag}
                <span class="tag">#{tag}</span>
              {/each}
              {#if memo.tags.length > 3}
                <span class="more-tags">+{memo.tags.length - 3}</span>
              {/if}
            </div>
          {/if}
        </button>
      {/each}
    {/if}
  </div>
</div>

<!-- Memo Editor Modal -->
<Modal
  open={showEditor}
  size="lg"
  title=""
  onClose={() => {
    showEditor = false;
    editingMemo = null;
  }}
>
  <MemoEditor
    memo={editingMemo}
    {codes}
    {linkedDataPoints}
    onSave={handleSave}
    onCancel={() => {
      showEditor = false;
      editingMemo = null;
    }}
    onDelete={editingMemo ? handleDelete : undefined}
  />
</Modal>

<style>
  .memo-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: white;
    border-radius: 0.5rem;
    overflow: hidden;
  }

  :global(.dark) .memo-panel {
    background: #1e293b;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #e2e8f0;
  }

  :global(.dark) .panel-header {
    border-color: #475569;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .new-button {
    padding: 0.375rem;
    color: #3b82f6;
    background: #eff6ff;
    border: none;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  :global(.dark) .new-button {
    background: #1e3a5f;
  }

  .new-button:hover {
    background: #dbeafe;
  }

  :global(.dark) .new-button:hover {
    background: #1d4ed8;
  }

  .filters {
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
  }

  :global(.dark) .filters {
    border-color: #475569;
  }

  .search-box {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.375rem 0.5rem;
    background: #f1f5f9;
    border-radius: 0.375rem;
  }

  :global(.dark) .search-box {
    background: #334155;
  }

  .search-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    font-size: 0.875rem;
    color: inherit;
  }

  .clear-search {
    padding: 0.125rem;
    color: #64748b;
    background: transparent;
    border: none;
    cursor: pointer;
  }

  .filter-row {
    display: flex;
    gap: 0.5rem;
  }

  .filter-select {
    flex: 1;
    padding: 0.375rem 0.5rem;
    font-size: 0.75rem;
    color: #475569;
    background: #f1f5f9;
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
  }

  :global(.dark) .filter-select {
    background: #334155;
    color: #cbd5e1;
  }

  .memo-list {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 2rem;
    text-align: center;
  }

  .memo-card {
    width: 100%;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
  }

  :global(.dark) .memo-card {
    background: #334155;
    border-color: #475569;
  }

  .memo-card:hover {
    background: #f1f5f9;
    border-color: #cbd5e1;
  }

  :global(.dark) .memo-card:hover {
    background: #475569;
    border-color: #64748b;
  }

  .memo-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .memo-type-badge {
    padding: 0.125rem 0.375rem;
    font-size: 0.625rem;
    font-weight: 500;
    color: white;
    border-radius: 9999px;
    text-transform: capitalize;
  }

  .memo-date {
    font-size: 0.625rem;
    color: #94a3b8;
  }

  .memo-content {
    font-size: 0.875rem;
    color: #475569;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }

  :global(.dark) .memo-content {
    color: #cbd5e1;
  }

  .memo-codes {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-bottom: 0.375rem;
  }

  .code-badge {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.125rem 0.375rem;
    font-size: 0.625rem;
    color: #64748b;
    background: white;
    border-radius: 9999px;
  }

  :global(.dark) .code-badge {
    background: #1e293b;
    color: #94a3b8;
  }

  .code-dot {
    width: 0.375rem;
    height: 0.375rem;
    border-radius: 50%;
  }

  .more-codes,
  .more-tags {
    font-size: 0.625rem;
    color: #94a3b8;
  }

  .memo-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
  }

  .tag {
    font-size: 0.625rem;
    color: #6366f1;
  }

  :global(.dark) .tag {
    color: #a5b4fc;
  }
</style>
