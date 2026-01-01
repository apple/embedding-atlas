<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconClose, IconPlus } from "../assets/icons.js";
  import type { ChartContext, RowID } from "../charts/chart.js";
  import ContentRenderer from "../renderers/ContentRenderer.svelte";
  import { isImage, isLink, type ColumnStyle } from "../renderers/index.js";
  import { imageToDataUrl } from "../utils/image.js";
  import Modal from "../widgets/Modal.svelte";
  import type { CodingStore } from "./store.svelte.js";
  import type { Code } from "./types.js";

  interface Props {
    open: boolean;
    dataPointId: RowID | null;
    data: Record<string, any> | null;
    context: ChartContext;
    codingStore: CodingStore;
    columnStyles: Record<string, ColumnStyle>;
    visibleColumns: string[];
    onClose: () => void;
    onNavigate?: (direction: "prev" | "next") => void;
  }

  let {
    open,
    dataPointId,
    data,
    context,
    codingStore,
    columnStyles,
    visibleColumns,
    onClose,
    onNavigate,
  }: Props = $props();

  let appliedCodes = $derived(
    dataPointId ? codingStore.getCodesForDataPoint(dataPointId) : []
  );

  let newCodeName = $state("");
  let showNewCodeInput = $state(false);

  function handleCreateCode() {
    if (newCodeName.trim() && dataPointId) {
      const code = codingStore.createCode(newCodeName.trim());
      codingStore.applyCode(code.id, [dataPointId]);
      newCodeName = "";
      showNewCodeInput = false;
    }
  }

  function handleToggleCode(code: Code) {
    if (dataPointId) {
      codingStore.toggleCode(code.id, dataPointId);
    }
  }

  function handleRemoveCode(codeId: string) {
    if (dataPointId) {
      codingStore.removeCode(codeId, [dataPointId]);
    }
  }

  function getDisplayValue(value: any): string {
    if (value === null || value === undefined) return "(empty)";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function isImageValue(value: any): boolean {
    return isImage(value) || (typeof value === "string" && /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(value));
  }

  function isVideoValue(value: any): boolean {
    if (typeof value !== "string") return false;
    return /\.(mp4|webm|ogg|mov)$/i.test(value) ||
           /youtube\.com|youtu\.be|vimeo\.com/i.test(value);
  }

  function handleKeydown(e: KeyboardEvent) {
    if (!open) return;
    if (e.key === "ArrowLeft" && onNavigate) {
      onNavigate("prev");
      e.preventDefault();
    } else if (e.key === "ArrowRight" && onNavigate) {
      onNavigate("next");
      e.preventDefault();
    }
  }

  // Find the primary content (image, video, or text)
  let primaryContent = $derived.by(() => {
    if (!data) return null;

    // First, look for image columns
    for (const col of visibleColumns) {
      const value = data[col];
      if (isImageValue(value)) {
        return { type: "image" as const, column: col, value };
      }
    }

    // Then, look for video columns
    for (const col of visibleColumns) {
      const value = data[col];
      if (isVideoValue(value)) {
        return { type: "video" as const, column: col, value };
      }
    }

    // Then, look for long text
    for (const col of visibleColumns) {
      const value = data[col];
      if (typeof value === "string" && value.length > 100) {
        return { type: "text" as const, column: col, value };
      }
    }

    return null;
  });

  let availableCodes = $derived(
    Object.values(codingStore.codes).filter(
      (code) => !appliedCodes.some((ac) => ac.id === code.id)
    )
  );
</script>

<svelte:window onkeydown={handleKeydown} />

<Modal {open} size="xl" title="Data Point Details" onClose={onClose}>
  {#snippet header()}
    <div class="flex items-center gap-4">
      <h2 class="text-lg font-semibold text-slate-800 dark:text-slate-200">
        Data Point Details
      </h2>
      {#if dataPointId !== null}
        <span class="text-sm text-slate-500 dark:text-slate-400">
          ID: {dataPointId}
        </span>
      {/if}
      {#if onNavigate}
        <div class="flex gap-1 ml-auto mr-4">
          <button
            onclick={() => onNavigate?.("prev")}
            class="px-2 py-1 text-sm rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600"
            title="Previous (←)"
          >
            ← Prev
          </button>
          <button
            onclick={() => onNavigate?.("next")}
            class="px-2 py-1 text-sm rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600"
            title="Next (→)"
          >
            Next →
          </button>
        </div>
      {/if}
    </div>
  {/snippet}

  {#if data}
    <div class="flex flex-col lg:flex-row gap-6">
      <!-- Left: Primary Content Preview -->
      <div class="flex-1 min-w-0">
        {#if primaryContent}
          <div class="mb-4">
            <h3 class="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2">
              {primaryContent.column}
            </h3>
            {#if primaryContent.type === "image"}
              <div class="flex justify-center bg-slate-100 dark:bg-slate-800 rounded-lg p-4">
                <img
                  src={isImage(primaryContent.value) ? imageToDataUrl(primaryContent.value) : primaryContent.value}
                  alt=""
                  class="max-w-full max-h-96 object-contain rounded"
                  referrerpolicy="no-referrer"
                />
              </div>
            {:else if primaryContent.type === "video"}
              <div class="flex justify-center bg-slate-100 dark:bg-slate-800 rounded-lg p-4">
                {#if /youtube\.com|youtu\.be/i.test(primaryContent.value)}
                  {@const videoId = primaryContent.value.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\s]+)/)?.[1]}
                  {#if videoId}
                    <iframe
                      width="100%"
                      height="315"
                      src="https://www.youtube.com/embed/{videoId}"
                      title="YouTube video player"
                      frameborder="0"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowfullscreen
                      class="rounded"
                    ></iframe>
                  {/if}
                {:else}
                  <video
                    src={primaryContent.value}
                    controls
                    class="max-w-full max-h-96 rounded"
                  >
                    <track kind="captions" />
                  </video>
                {/if}
              </div>
            {:else}
              <div class="bg-slate-100 dark:bg-slate-800 rounded-lg p-4 max-h-64 overflow-auto">
                <ContentRenderer
                  value={primaryContent.value}
                  renderer={columnStyles[primaryContent.column]?.renderer}
                />
              </div>
            {/if}
          </div>
        {/if}

        <!-- Metadata Table -->
        <div class="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
          <table class="w-full text-sm">
            <thead class="bg-slate-100 dark:bg-slate-800">
              <tr>
                <th class="px-3 py-2 text-left font-medium text-slate-600 dark:text-slate-300 w-1/3">
                  Column
                </th>
                <th class="px-3 py-2 text-left font-medium text-slate-600 dark:text-slate-300">
                  Value
                </th>
              </tr>
            </thead>
            <tbody>
              {#each visibleColumns as column}
                {@const value = data[column]}
                {@const style = columnStyles[column]}
                {#if style?.display !== "hidden" && column !== primaryContent?.column}
                  <tr class="border-t border-slate-200 dark:border-slate-700">
                    <td class="px-3 py-2 font-medium text-slate-600 dark:text-slate-400 align-top">
                      {column}
                    </td>
                    <td class="px-3 py-2 text-slate-800 dark:text-slate-200">
                      {#if isImageValue(value)}
                        <img
                          src={isImage(value) ? imageToDataUrl(value) : value}
                          alt=""
                          class="max-w-32 max-h-32 object-contain rounded"
                          referrerpolicy="no-referrer"
                        />
                      {:else if isLink(value)}
                        <a href={value} target="_blank" class="text-blue-600 dark:text-blue-400 underline break-all">
                          {value}
                        </a>
                      {:else}
                        <ContentRenderer
                          value={value}
                          renderer={style?.renderer}
                        />
                      {/if}
                    </td>
                  </tr>
                {/if}
              {/each}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Right: Coding Panel -->
      <div class="w-full lg:w-80 flex-shrink-0">
        <div class="sticky top-0">
          <h3 class="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
            Applied Codes
          </h3>

          <!-- Applied codes -->
          <div class="flex flex-wrap gap-2 mb-4 min-h-[2rem]">
            {#if appliedCodes.length === 0}
              <span class="text-sm text-slate-400 dark:text-slate-500 italic">
                No codes applied
              </span>
            {:else}
              {#each appliedCodes as code}
                <span
                  class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm text-white"
                  style:background-color={code.color}
                >
                  {code.name}
                  <button
                    onclick={() => handleRemoveCode(code.id)}
                    class="ml-1 hover:bg-white/20 rounded-full p-0.5"
                    title="Remove code"
                  >
                    <IconClose class="w-3 h-3" />
                  </button>
                </span>
              {/each}
            {/if}
          </div>

          <!-- Available codes -->
          <h3 class="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
            Available Codes
          </h3>

          <div class="space-y-1 max-h-64 overflow-y-auto mb-4">
            {#each availableCodes as code}
              <button
                onclick={() => handleToggleCode(code)}
                class="w-full flex items-center gap-2 px-3 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-left transition-colors"
              >
                <span
                  class="w-3 h-3 rounded-full flex-shrink-0"
                  style:background-color={code.color}
                ></span>
                <span class="text-sm text-slate-700 dark:text-slate-300 truncate">
                  {code.name}
                </span>
                {#if code.frequency > 0}
                  <span class="ml-auto text-xs text-slate-400">
                    {code.frequency}
                  </span>
                {/if}
              </button>
            {/each}
          </div>

          <!-- Create new code -->
          <div class="border-t border-slate-200 dark:border-slate-700 pt-4">
            {#if showNewCodeInput}
              <div class="flex gap-2">
                <input
                  type="text"
                  bind:value={newCodeName}
                  placeholder="Code name..."
                  class="flex-1 px-3 py-2 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  onkeydown={(e) => {
                    if (e.key === "Enter") handleCreateCode();
                    if (e.key === "Escape") showNewCodeInput = false;
                  }}
                />
                <button
                  onclick={handleCreateCode}
                  disabled={!newCodeName.trim()}
                  class="px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Add
                </button>
              </div>
            {:else}
              <button
                onclick={() => (showNewCodeInput = true)}
                class="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
              >
                <IconPlus class="w-4 h-4" />
                Create new code
              </button>
            {/if}
          </div>
        </div>
      </div>
    </div>
  {:else}
    <div class="text-center py-8 text-slate-500 dark:text-slate-400">
      No data point selected
    </div>
  {/if}
</Modal>
