<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconClose, IconTag } from "../assets/icons.js";
  import type { Memo, Code } from "./types.js";
  import type { RowID } from "../charts/chart.js";

  interface Props {
    memo?: Memo | null;
    codes: Record<string, Code>;
    linkedDataPoints?: RowID[];
    onSave: (memo: Omit<Memo, "id" | "createdAt">) => void;
    onCancel: () => void;
    onDelete?: () => void;
  }

  let {
    memo = null,
    codes,
    linkedDataPoints = [],
    onSave,
    onCancel,
    onDelete,
  }: Props = $props();

  // Form state
  let content = $state(memo?.content ?? "");
  let memoType = $state<Memo["memoType"]>(memo?.memoType ?? "observational");
  let selectedCodes = $state<string[]>(memo?.linkedCodes ?? []);
  let tags = $state<string[]>(memo?.tags ?? []);
  let newTag = $state("");

  const memoTypes: { value: Memo["memoType"]; label: string; description: string }[] = [
    {
      value: "theoretical",
      label: "Theoretical",
      description: "Ideas about relationships between concepts and emerging theory",
    },
    {
      value: "methodological",
      label: "Methodological",
      description: "Notes about research process, decisions, and procedures",
    },
    {
      value: "observational",
      label: "Observational",
      description: "Descriptive notes about what you observe in the data",
    },
  ];

  function handleSave() {
    if (!content.trim()) return;

    onSave({
      content: content.trim(),
      memoType,
      linkedCodes: selectedCodes,
      linkedDataPoints: linkedDataPoints,
      tags,
      createdBy: undefined,
    });
  }

  function toggleCode(codeId: string) {
    if (selectedCodes.includes(codeId)) {
      selectedCodes = selectedCodes.filter((id) => id !== codeId);
    } else {
      selectedCodes = [...selectedCodes, codeId];
    }
  }

  function addTag() {
    const tag = newTag.trim().toLowerCase();
    if (tag && !tags.includes(tag)) {
      tags = [...tags, tag];
      newTag = "";
    }
  }

  function removeTag(tag: string) {
    tags = tags.filter((t) => t !== tag);
  }

  let isValid = $derived(content.trim().length > 0);
  let codeList = $derived(Object.values(codes));
</script>

<div class="memo-editor">
  <div class="editor-header">
    <h3 class="text-lg font-semibold text-slate-800 dark:text-slate-200">
      {memo ? "Edit Memo" : "New Memo"}
    </h3>
    <button onclick={onCancel} class="close-button" title="Cancel">
      <IconClose class="w-5 h-5" />
    </button>
  </div>

  <div class="editor-content">
    <!-- Memo Type -->
    <div class="form-group">
      <label class="form-label">Memo Type</label>
      <div class="type-options">
        {#each memoTypes as type}
          <button
            class="type-option"
            class:selected={memoType === type.value}
            onclick={() => (memoType = type.value)}
            title={type.description}
          >
            {type.label}
          </button>
        {/each}
      </div>
    </div>

    <!-- Content -->
    <div class="form-group">
      <label class="form-label" for="memo-content">Content</label>
      <textarea
        id="memo-content"
        bind:value={content}
        placeholder="Write your memo here..."
        class="content-textarea"
        rows="8"
      ></textarea>
    </div>

    <!-- Linked Codes -->
    <div class="form-group">
      <label class="form-label">Link to Codes (optional)</label>
      <div class="linked-codes">
        {#if codeList.length === 0}
          <span class="text-sm text-slate-400 italic">No codes available</span>
        {:else}
          {#each codeList as code}
            <button
              class="code-chip"
              class:selected={selectedCodes.includes(code.id)}
              onclick={() => toggleCode(code.id)}
            >
              <span
                class="code-dot"
                style:background-color={code.color}
              ></span>
              {code.name}
            </button>
          {/each}
        {/if}
      </div>
    </div>

    <!-- Tags -->
    <div class="form-group">
      <label class="form-label">Tags (optional)</label>
      <div class="tags-container">
        {#each tags as tag}
          <span class="tag">
            #{tag}
            <button onclick={() => removeTag(tag)} class="tag-remove">
              <IconClose class="w-3 h-3" />
            </button>
          </span>
        {/each}
        <input
          type="text"
          bind:value={newTag}
          placeholder="Add tag..."
          class="tag-input"
          onkeydown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              addTag();
            }
          }}
        />
      </div>
    </div>

    <!-- Linked Data Points Info -->
    {#if linkedDataPoints.length > 0}
      <div class="info-box">
        <IconTag class="w-4 h-4 text-blue-500" />
        <span class="text-sm text-slate-600 dark:text-slate-400">
          This memo will be linked to {linkedDataPoints.length} data point{linkedDataPoints.length > 1 ? "s" : ""}
        </span>
      </div>
    {/if}
  </div>

  <div class="editor-footer">
    {#if memo && onDelete}
      <button onclick={onDelete} class="delete-button">
        Delete Memo
      </button>
    {/if}
    <div class="footer-actions">
      <button onclick={onCancel} class="cancel-button">
        Cancel
      </button>
      <button
        onclick={handleSave}
        disabled={!isValid}
        class="save-button"
      >
        {memo ? "Update" : "Create"} Memo
      </button>
    </div>
  </div>
</div>

<style>
  .memo-editor {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: white;
    border-radius: 0.5rem;
    overflow: hidden;
  }

  :global(.dark) .memo-editor {
    background: #1e293b;
  }

  .editor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-bottom: 1px solid #e2e8f0;
  }

  :global(.dark) .editor-header {
    border-color: #475569;
  }

  .close-button {
    padding: 0.25rem;
    color: #64748b;
    background: transparent;
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
  }

  .close-button:hover {
    background: #f1f5f9;
    color: #334155;
  }

  :global(.dark) .close-button:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .editor-content {
    flex: 1;
    padding: 1rem;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .form-label {
    font-size: 0.875rem;
    font-weight: 500;
    color: #475569;
  }

  :global(.dark) .form-label {
    color: #cbd5e1;
  }

  .type-options {
    display: flex;
    gap: 0.5rem;
  }

  .type-option {
    padding: 0.375rem 0.75rem;
    font-size: 0.875rem;
    color: #64748b;
    background: #f1f5f9;
    border: 1px solid transparent;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  :global(.dark) .type-option {
    background: #334155;
    color: #94a3b8;
  }

  .type-option:hover {
    background: #e2e8f0;
  }

  :global(.dark) .type-option:hover {
    background: #475569;
  }

  .type-option.selected {
    background: #3b82f6;
    color: white;
    border-color: #3b82f6;
  }

  .content-textarea {
    width: 100%;
    padding: 0.75rem;
    font-size: 0.875rem;
    line-height: 1.5;
    color: #1e293b;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    resize: vertical;
  }

  :global(.dark) .content-textarea {
    color: #e2e8f0;
    background: #0f172a;
    border-color: #475569;
  }

  .content-textarea:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .linked-codes {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
  }

  .code-chip {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    color: #475569;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 9999px;
    cursor: pointer;
    transition: all 0.15s;
  }

  :global(.dark) .code-chip {
    color: #cbd5e1;
    background: #334155;
    border-color: #475569;
  }

  .code-chip:hover {
    background: #e2e8f0;
  }

  :global(.dark) .code-chip:hover {
    background: #475569;
  }

  .code-chip.selected {
    background: #dbeafe;
    border-color: #3b82f6;
    color: #1d4ed8;
  }

  :global(.dark) .code-chip.selected {
    background: #1e3a5f;
    color: #93c5fd;
  }

  .code-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
  }

  .tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
    align-items: center;
  }

  .tag {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.125rem 0.5rem;
    font-size: 0.75rem;
    color: #6366f1;
    background: #eef2ff;
    border-radius: 9999px;
  }

  :global(.dark) .tag {
    background: #312e81;
    color: #a5b4fc;
  }

  .tag-remove {
    padding: 0;
    background: transparent;
    border: none;
    cursor: pointer;
    opacity: 0.6;
  }

  .tag-remove:hover {
    opacity: 1;
  }

  .tag-input {
    flex: 1;
    min-width: 80px;
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    background: transparent;
    border: none;
    outline: none;
    color: inherit;
  }

  .info-box {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    background: #eff6ff;
    border-radius: 0.375rem;
  }

  :global(.dark) .info-box {
    background: #1e3a5f;
  }

  .editor-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-top: 1px solid #e2e8f0;
  }

  :global(.dark) .editor-footer {
    border-color: #475569;
  }

  .footer-actions {
    display: flex;
    gap: 0.5rem;
    margin-left: auto;
  }

  .delete-button {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    color: #dc2626;
    background: transparent;
    border: 1px solid #fecaca;
    border-radius: 0.375rem;
    cursor: pointer;
  }

  .delete-button:hover {
    background: #fef2f2;
  }

  :global(.dark) .delete-button {
    border-color: #7f1d1d;
  }

  :global(.dark) .delete-button:hover {
    background: #450a0a;
  }

  .cancel-button {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    color: #64748b;
    background: transparent;
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    cursor: pointer;
  }

  :global(.dark) .cancel-button {
    border-color: #475569;
    color: #94a3b8;
  }

  .cancel-button:hover {
    background: #f1f5f9;
  }

  :global(.dark) .cancel-button:hover {
    background: #334155;
  }

  .save-button {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    font-weight: 500;
    color: white;
    background: #3b82f6;
    border: none;
    border-radius: 0.375rem;
    cursor: pointer;
  }

  .save-button:hover:not(:disabled) {
    background: #2563eb;
  }

  .save-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
