<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconDownload } from "../assets/icons.js";
  import Select from "../widgets/Select.svelte";
  import type { CodingState } from "./types.js";
  import {
    exportAndDownload,
    exportCodingState,
    exportCodebook,
    exportMemos,
    generateCodingSummary,
    downloadAsFile,
    type ExportFormat,
    type ExportOptions,
    type DataRow,
  } from "./data_export.js";

  interface Props {
    rows: DataRow[];
    columns: string[];
    codingState: CodingState | null;
    selectedRows?: Set<number | string>;
  }

  let {
    rows,
    columns,
    codingState,
    selectedRows,
  }: Props = $props();

  let format = $state<ExportFormat>("csv");
  let includeCoding = $state(true);
  let includeMemos = $state(true);
  let includeCodeMetadata = $state(false);
  let exportOnlySelected = $state(false);
  let filename = $state("export");

  const formatOptions = [
    { value: "csv", label: "CSV" },
    { value: "json", label: "JSON" },
    { value: "jsonl", label: "JSON Lines" },
  ];

  function handleExportData() {
    const options: ExportOptions = {
      includeCoding,
      includeMemos,
      includeCodeMetadata,
      selectedRows: exportOnlySelected && selectedRows ? selectedRows : undefined,
    };

    exportAndDownload(rows, columns, codingState, format, filename, options);
  }

  function handleExportCodingState() {
    if (!codingState) return;
    const content = exportCodingState(codingState);
    downloadAsFile(content, "coding_state.json", "application/json");
  }

  function handleExportCodebook() {
    if (!codingState) return;
    const content = exportCodebook(codingState);
    downloadAsFile(content, "codebook.json", "application/json");
  }

  function handleExportMemos() {
    if (!codingState) return;
    const content = exportMemos(codingState);
    downloadAsFile(content, "memos.json", "application/json");
  }

  function handleExportSummary() {
    if (!codingState) return;
    const content = generateCodingSummary(codingState, rows.length);
    downloadAsFile(content, "coding_summary.json", "application/json");
  }

  let hasSelection = $derived(selectedRows && selectedRows.size > 0);
  let hasCoding = $derived(codingState && Object.keys(codingState.codes).length > 0);
  let hasMemos = $derived(codingState && Object.keys(codingState.memos).length > 0);
</script>

<div class="export-panel">
  <h4 class="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
    Export Data
  </h4>

  <!-- Main export options -->
  <div class="export-section">
    <div class="form-group">
      <label>Filename</label>
      <input
        type="text"
        bind:value={filename}
        class="form-input"
        placeholder="export"
      />
    </div>

    <div class="form-group">
      <label>Format</label>
      <Select
        options={formatOptions}
        value={format}
        onChange={(v) => (format = v as ExportFormat)}
      />
    </div>

    <div class="checkbox-group">
      <label class="checkbox-label">
        <input
          type="checkbox"
          bind:checked={includeCoding}
          disabled={!hasCoding}
        />
        <span>Include codes</span>
      </label>

      <label class="checkbox-label">
        <input
          type="checkbox"
          bind:checked={includeMemos}
          disabled={!hasMemos}
        />
        <span>Include memos</span>
      </label>

      <label class="checkbox-label">
        <input
          type="checkbox"
          bind:checked={includeCodeMetadata}
          disabled={!includeCoding || !hasCoding}
        />
        <span>Include code metadata</span>
      </label>

      {#if hasSelection}
        <label class="checkbox-label">
          <input
            type="checkbox"
            bind:checked={exportOnlySelected}
          />
          <span>Export only selected ({selectedRows?.size} rows)</span>
        </label>
      {/if}
    </div>

    <button
      onclick={handleExportData}
      class="export-button primary"
    >
      <IconDownload class="w-4 h-4" />
      Export Data
    </button>
  </div>

  <!-- Additional exports -->
  {#if hasCoding || hasMemos}
    <div class="divider"></div>

    <h5 class="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">
      Additional Exports
    </h5>

    <div class="additional-exports">
      {#if hasCoding}
        <button
          onclick={handleExportCodebook}
          class="export-button secondary"
        >
          <IconDownload class="w-3 h-3" />
          Codebook
        </button>

        <button
          onclick={handleExportSummary}
          class="export-button secondary"
        >
          <IconDownload class="w-3 h-3" />
          Summary Report
        </button>
      {/if}

      {#if hasMemos}
        <button
          onclick={handleExportMemos}
          class="export-button secondary"
        >
          <IconDownload class="w-3 h-3" />
          Memos Only
        </button>
      {/if}

      <button
        onclick={handleExportCodingState}
        class="export-button secondary"
        disabled={!hasCoding && !hasMemos}
      >
        <IconDownload class="w-3 h-3" />
        Full Coding State
      </button>
    </div>
  {/if}
</div>

<style>
  .export-panel {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .export-section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .form-group label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #64748b;
  }

  :global(.dark) .form-group label {
    color: #94a3b8;
  }

  .form-input {
    padding: 0.375rem 0.5rem;
    font-size: 0.875rem;
    border: 1px solid #cbd5e1;
    border-radius: 0.25rem;
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
  }

  .checkbox-group {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: #475569;
    cursor: pointer;
  }

  :global(.dark) .checkbox-label {
    color: #cbd5e1;
  }

  .checkbox-label input:disabled + span {
    opacity: 0.5;
  }

  .export-button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.375rem;
    padding: 0.5rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 500;
    border-radius: 0.25rem;
    cursor: pointer;
    transition: all 0.15s;
  }

  .export-button.primary {
    background: #3b82f6;
    color: white;
    border: none;
  }

  .export-button.primary:hover {
    background: #2563eb;
  }

  .export-button.secondary {
    background: transparent;
    color: #475569;
    border: 1px solid #cbd5e1;
  }

  :global(.dark) .export-button.secondary {
    border-color: #475569;
    color: #cbd5e1;
  }

  .export-button.secondary:hover {
    background: #f1f5f9;
  }

  :global(.dark) .export-button.secondary:hover {
    background: #334155;
  }

  .export-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .divider {
    height: 1px;
    background: #e2e8f0;
    margin: 0.5rem 0;
  }

  :global(.dark) .divider {
    background: #475569;
  }

  .additional-exports {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
</style>
