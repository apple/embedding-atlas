<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconPlus, IconClose } from "../assets/icons.js";
  import type { Code } from "./types.js";
  import {
    type CodeRelationship,
    type RelationshipType,
    relationshipTypes,
    getCodeRelationships,
    findCoOccurrences,
  } from "./code_relationships.js";
  import { generateId } from "./types.js";

  interface Props {
    code: Code;
    codes: Record<string, Code>;
    relationships: CodeRelationship[];
    codeApplications: Record<string, string[]>;
    onAddRelationship: (rel: CodeRelationship) => void;
    onRemoveRelationship: (relId: string) => void;
  }

  let {
    code,
    codes,
    relationships,
    codeApplications,
    onAddRelationship,
    onRemoveRelationship,
  }: Props = $props();

  let showAddForm = $state(false);
  let selectedTargetId = $state<string>("");
  let selectedType = $state<RelationshipType>("associated");
  let description = $state("");

  // Get relationships for this code
  let codeRelationships = $derived(
    getCodeRelationships(relationships, code.id)
  );

  // Find suggested co-occurrences
  let coOccurrences = $derived(
    findCoOccurrences(codeApplications, 2)
      .filter(co =>
        (co.codeA === code.id || co.codeB === code.id) &&
        !relationships.some(r =>
          (r.sourceCodeId === co.codeA && r.targetCodeId === co.codeB) ||
          (r.sourceCodeId === co.codeB && r.targetCodeId === co.codeA)
        )
      )
      .slice(0, 5)
  );

  // Available codes (excluding self)
  let availableCodes = $derived(
    Object.values(codes).filter(c => c.id !== code.id)
  );

  function handleAddRelationship() {
    if (!selectedTargetId) return;

    onAddRelationship({
      id: generateId(),
      sourceCodeId: code.id,
      targetCodeId: selectedTargetId,
      type: selectedType,
      description: description.trim() || undefined,
      createdAt: Date.now(),
    });

    showAddForm = false;
    selectedTargetId = "";
    selectedType = "associated";
    description = "";
  }

  function suggestRelationship(codeA: string, codeB: string, count: number) {
    const targetId = codeA === code.id ? codeB : codeA;
    selectedTargetId = targetId;
    selectedType = "co_occurs";
    description = `Suggested: co-occurs ${count} times`;
    showAddForm = true;
  }

  function getRelationshipLabel(rel: CodeRelationship): string {
    const typeInfo = relationshipTypes[rel.type];
    const otherCodeId = rel.sourceCodeId === code.id ? rel.targetCodeId : rel.sourceCodeId;
    const otherCode = codes[otherCodeId];
    const direction = rel.sourceCodeId === code.id ? "→" : "←";

    if (typeInfo.directed) {
      return `${direction} ${typeInfo.label}: ${otherCode?.name || "Unknown"}`;
    }
    return `${typeInfo.label}: ${otherCode?.name || "Unknown"}`;
  }
</script>

<div class="relationship-editor">
  <div class="header">
    <h4 class="text-sm font-semibold text-slate-700 dark:text-slate-300">
      Relationships
    </h4>
    {#if !showAddForm}
      <button
        onclick={() => (showAddForm = true)}
        class="add-button"
        title="Add relationship"
      >
        <IconPlus class="w-4 h-4" />
      </button>
    {/if}
  </div>

  <!-- Add Relationship Form -->
  {#if showAddForm}
    <div class="add-form">
      <div class="form-row">
        <select
          bind:value={selectedTargetId}
          class="form-select"
        >
          <option value="">Select code...</option>
          {#each availableCodes as targetCode}
            <option value={targetCode.id}>{targetCode.name}</option>
          {/each}
        </select>
      </div>

      <div class="form-row">
        <select
          bind:value={selectedType}
          class="form-select"
        >
          {#each Object.entries(relationshipTypes) as [type, info]}
            <option value={type}>{info.label}</option>
          {/each}
        </select>
      </div>

      <div class="form-row">
        <input
          type="text"
          bind:value={description}
          placeholder="Description (optional)"
          class="form-input"
        />
      </div>

      <div class="form-actions">
        <button
          onclick={() => {
            showAddForm = false;
            selectedTargetId = "";
            description = "";
          }}
          class="btn-cancel"
        >
          Cancel
        </button>
        <button
          onclick={handleAddRelationship}
          disabled={!selectedTargetId}
          class="btn-add"
        >
          Add
        </button>
      </div>
    </div>
  {/if}

  <!-- Existing Relationships -->
  <div class="relationships-list">
    {#if codeRelationships.outgoing.length === 0 &&
        codeRelationships.incoming.length === 0 &&
        codeRelationships.bidirectional.length === 0}
      <p class="empty-message">No relationships defined</p>
    {:else}
      <!-- Outgoing -->
      {#each codeRelationships.outgoing as rel}
        {@const typeInfo = relationshipTypes[rel.type]}
        {@const targetCode = codes[rel.targetCodeId]}
        <div class="relationship-item">
          <span
            class="rel-type"
            style:background-color={typeInfo.color}
          >
            → {typeInfo.label}
          </span>
          <span class="rel-target">
            <span
              class="code-dot"
              style:background-color={targetCode?.color}
            ></span>
            {targetCode?.name || "Unknown"}
          </span>
          <button
            onclick={() => onRemoveRelationship(rel.id)}
            class="remove-btn"
            title="Remove relationship"
          >
            <IconClose class="w-3 h-3" />
          </button>
        </div>
      {/each}

      <!-- Incoming -->
      {#each codeRelationships.incoming as rel}
        {@const typeInfo = relationshipTypes[rel.type]}
        {@const sourceCode = codes[rel.sourceCodeId]}
        <div class="relationship-item">
          <span
            class="rel-type"
            style:background-color={typeInfo.color}
          >
            ← {typeInfo.label}
          </span>
          <span class="rel-target">
            <span
              class="code-dot"
              style:background-color={sourceCode?.color}
            ></span>
            {sourceCode?.name || "Unknown"}
          </span>
          <button
            onclick={() => onRemoveRelationship(rel.id)}
            class="remove-btn"
            title="Remove relationship"
          >
            <IconClose class="w-3 h-3" />
          </button>
        </div>
      {/each}

      <!-- Bidirectional -->
      {#each codeRelationships.bidirectional as rel}
        {@const typeInfo = relationshipTypes[rel.type]}
        {@const otherCodeId = rel.sourceCodeId === code.id ? rel.targetCodeId : rel.sourceCodeId}
        {@const otherCode = codes[otherCodeId]}
        <div class="relationship-item">
          <span
            class="rel-type"
            style:background-color={typeInfo.color}
          >
            ↔ {typeInfo.label}
          </span>
          <span class="rel-target">
            <span
              class="code-dot"
              style:background-color={otherCode?.color}
            ></span>
            {otherCode?.name || "Unknown"}
          </span>
          <button
            onclick={() => onRemoveRelationship(rel.id)}
            class="remove-btn"
            title="Remove relationship"
          >
            <IconClose class="w-3 h-3" />
          </button>
        </div>
      {/each}
    {/if}
  </div>

  <!-- Suggested Co-occurrences -->
  {#if coOccurrences.length > 0}
    <div class="suggestions">
      <h5 class="text-xs text-slate-500 dark:text-slate-400 mb-2">
        Suggested (co-occurrence)
      </h5>
      {#each coOccurrences as co}
        {@const otherCodeId = co.codeA === code.id ? co.codeB : co.codeA}
        {@const otherCode = codes[otherCodeId]}
        {#if otherCode}
          <button
            class="suggestion-item"
            onclick={() => suggestRelationship(co.codeA, co.codeB, co.count)}
          >
            <span
              class="code-dot"
              style:background-color={otherCode.color}
            ></span>
            <span class="suggestion-name">{otherCode.name}</span>
            <span class="suggestion-count">{co.count}x</span>
          </button>
        {/if}
      {/each}
    </div>
  {/if}
</div>

<style>
  .relationship-editor {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .add-button {
    padding: 0.25rem;
    color: #3b82f6;
    background: #eff6ff;
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
  }

  :global(.dark) .add-button {
    background: #1e3a5f;
  }

  .add-button:hover {
    background: #dbeafe;
  }

  .add-form {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.5rem;
    background: #f1f5f9;
    border-radius: 0.375rem;
  }

  :global(.dark) .add-form {
    background: #334155;
  }

  .form-row {
    display: flex;
    gap: 0.5rem;
  }

  .form-select,
  .form-input {
    flex: 1;
    padding: 0.375rem 0.5rem;
    font-size: 0.75rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.25rem;
    background: white;
    color: #334155;
  }

  :global(.dark) .form-select,
  :global(.dark) .form-input {
    border-color: #475569;
    background: #1e293b;
    color: #e2e8f0;
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
  }

  .btn-cancel,
  .btn-add {
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    border-radius: 0.25rem;
    cursor: pointer;
  }

  .btn-cancel {
    color: #64748b;
    background: transparent;
    border: 1px solid #e2e8f0;
  }

  :global(.dark) .btn-cancel {
    border-color: #475569;
    color: #94a3b8;
  }

  .btn-add {
    color: white;
    background: #3b82f6;
    border: none;
  }

  .btn-add:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .relationships-list {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .empty-message {
    font-size: 0.75rem;
    color: #94a3b8;
    font-style: italic;
  }

  .relationship-item {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem;
    background: #f8fafc;
    border-radius: 0.25rem;
  }

  :global(.dark) .relationship-item {
    background: #334155;
  }

  .rel-type {
    padding: 0.125rem 0.375rem;
    font-size: 0.625rem;
    font-weight: 500;
    color: white;
    border-radius: 9999px;
  }

  .rel-target {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.75rem;
    color: #475569;
  }

  :global(.dark) .rel-target {
    color: #cbd5e1;
  }

  .code-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .remove-btn {
    padding: 0.125rem;
    color: #94a3b8;
    background: transparent;
    border: none;
    cursor: pointer;
  }

  .remove-btn:hover {
    color: #ef4444;
  }

  .suggestions {
    padding-top: 0.5rem;
    border-top: 1px dashed #e2e8f0;
  }

  :global(.dark) .suggestions {
    border-color: #475569;
  }

  .suggestion-item {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    width: 100%;
    padding: 0.25rem 0.375rem;
    background: transparent;
    border: 1px dashed #e2e8f0;
    border-radius: 0.25rem;
    cursor: pointer;
    margin-bottom: 0.25rem;
    text-align: left;
  }

  :global(.dark) .suggestion-item {
    border-color: #475569;
  }

  .suggestion-item:hover {
    background: #f1f5f9;
    border-style: solid;
  }

  :global(.dark) .suggestion-item:hover {
    background: #334155;
  }

  .suggestion-name {
    flex: 1;
    font-size: 0.75rem;
    color: #64748b;
  }

  :global(.dark) .suggestion-name {
    color: #94a3b8;
  }

  .suggestion-count {
    font-size: 0.625rem;
    color: #22c55e;
  }
</style>
