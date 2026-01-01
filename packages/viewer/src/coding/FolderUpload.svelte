<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconFolder, IconPlus } from "../assets/icons.js";
  import { importFolder, type FolderImportOptions, type ImportResult, defaultFolderImportOptions } from "./data_import.js";

  interface Props {
    onImport: (result: ImportResult) => void;
    options?: FolderImportOptions;
  }

  let {
    onImport,
    options = defaultFolderImportOptions,
  }: Props = $props();

  let isDragging = $state(false);
  let isLoading = $state(false);
  let folderInput: HTMLInputElement;
  let error = $state<string | null>(null);

  async function handleFiles(files: FileList | File[]) {
    if (files.length === 0) return;

    isLoading = true;
    error = null;

    try {
      const result = await importFolder(files, options);
      if (result.errors.length > 0 && result.rows.length === 0) {
        error = result.errors.join("; ");
      } else {
        onImport(result);
      }
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to import folder";
    } finally {
      isLoading = false;
    }
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    isDragging = true;
  }

  function handleDragLeave() {
    isDragging = false;
  }

  async function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragging = false;

    if (event.dataTransfer?.files) {
      await handleFiles(event.dataTransfer.files);
    }
  }

  async function handleSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      await handleFiles(input.files);
    }
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="folder-upload"
  class:dragging={isDragging}
  class:loading={isLoading}
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  ondrop={handleDrop}
>
  <button
    type="button"
    class="upload-button"
    onclick={() => folderInput.click()}
    disabled={isLoading}
  >
    {#if isLoading}
      <div class="spinner"></div>
      <span>Importing...</span>
    {:else}
      <IconFolder class="w-8 h-8 text-slate-400" />
      <span class="text-slate-600 dark:text-slate-400">
        {isDragging ? "Drop folder here" : "Click to select folder or drag & drop"}
      </span>
      <span class="text-sm text-slate-400 dark:text-slate-500">
        Supports images (JPEG, PNG, GIF, WebP) and videos (MP4, WebM)
      </span>
    {/if}
  </button>

  <input
    bind:this={folderInput}
    type="file"
    class="hidden"
    multiple
    onchange={handleSelect}
    {...{ webkitdirectory: true, directory: true } as any}
  />

  {#if error}
    <div class="error-message">
      {error}
    </div>
  {/if}
</div>

<style>
  .folder-upload {
    width: 100%;
  }

  .upload-button {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    width: 100%;
    padding: 2rem;
    border: 2px dashed #cbd5e1;
    border-radius: 0.5rem;
    background: #f8fafc;
    cursor: pointer;
    transition: all 0.2s;
  }

  :global(.dark) .upload-button {
    border-color: #475569;
    background: #1e293b;
  }

  .upload-button:hover:not(:disabled) {
    border-color: #94a3b8;
    background: #f1f5f9;
  }

  :global(.dark) .upload-button:hover:not(:disabled) {
    border-color: #64748b;
    background: #334155;
  }

  .dragging .upload-button {
    border-color: #3b82f6;
    background: #eff6ff;
  }

  :global(.dark) .dragging .upload-button {
    border-color: #3b82f6;
    background: #1e3a5f;
  }

  .upload-button:disabled {
    cursor: not-allowed;
    opacity: 0.7;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 3px solid #e2e8f0;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-message {
    margin-top: 0.5rem;
    padding: 0.5rem;
    color: #dc2626;
    background: #fef2f2;
    border-radius: 0.25rem;
    font-size: 0.875rem;
  }

  :global(.dark) .error-message {
    background: #450a0a;
    color: #fca5a5;
  }

  .hidden {
    display: none;
  }
</style>
