<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { slide } from "svelte/transition";

  import { IconClose, IconPlus } from "../assets/icons.js";
  import PopupButton from "../widgets/PopupButton.svelte";
  import type { CodingStore } from "./store.svelte.js";
  import type { SelectionStore } from "./selection.svelte.js";
  import type { Code } from "./types.js";

  interface Props {
    selectionStore: SelectionStore;
    codingStore: CodingStore;
    onViewDetails?: () => void;
    onClearSelection?: () => void;
  }

  let { selectionStore, codingStore, onViewDetails, onClearSelection }: Props = $props();

  let newCodeName = $state("");
  let showNewCodeInput = $state(false);

  let selectedCount = $derived(selectionStore.count);
  let selectedIds = $derived(selectionStore.selectedArray);

  // Get codes that are applied to ALL selected items
  let commonCodes = $derived.by(() => {
    if (selectedIds.length === 0) return [];

    const firstCodes = new Set(
      (codingStore.codeApplications[String(selectedIds[0])] ?? [])
    );

    for (let i = 1; i < selectedIds.length; i++) {
      const codes = new Set(
        codingStore.codeApplications[String(selectedIds[i])] ?? []
      );
      for (const codeId of firstCodes) {
        if (!codes.has(codeId)) {
          firstCodes.delete(codeId);
        }
      }
    }

    return Array.from(firstCodes)
      .map((id) => codingStore.codes[id])
      .filter(Boolean);
  });

  // Get codes that are applied to SOME selected items
  let partialCodes = $derived.by(() => {
    if (selectedIds.length === 0) return [];

    const allCodes = new Set<string>();
    const commonCodeIds = new Set(commonCodes.map((c) => c.id));

    for (const id of selectedIds) {
      const codes = codingStore.codeApplications[String(id)] ?? [];
      for (const codeId of codes) {
        if (!commonCodeIds.has(codeId)) {
          allCodes.add(codeId);
        }
      }
    }

    return Array.from(allCodes)
      .map((id) => codingStore.codes[id])
      .filter(Boolean);
  });

  let availableCodes = $derived(
    Object.values(codingStore.codes).filter(
      (code) =>
        !commonCodes.some((cc) => cc.id === code.id) &&
        !partialCodes.some((pc) => pc.id === code.id)
    )
  );

  function handleApplyCode(code: Code) {
    codingStore.applyCode(code.id, selectedIds);
  }

  function handleRemoveCode(code: Code) {
    codingStore.removeCode(code.id, selectedIds);
  }

  function handleCreateAndApplyCode() {
    if (newCodeName.trim() && selectedIds.length > 0) {
      const code = codingStore.createCode(newCodeName.trim());
      codingStore.applyCode(code.id, selectedIds);
      newCodeName = "";
      showNewCodeInput = false;
    }
  }

  function handleClear() {
    selectionStore.clear();
    onClearSelection?.();
  }
</script>

{#if selectedCount > 0}
  <div
    class="fixed bottom-4 left-1/2 -translate-x-1/2 z-40 flex items-center gap-3 px-4 py-3 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-lg shadow-xl"
    transition:slide={{ duration: 200 }}
  >
    <!-- Selection count -->
    <div class="flex items-center gap-2 pr-3 border-r border-slate-200 dark:border-slate-700">
      <span class="text-sm font-medium text-slate-700 dark:text-slate-300">
        {selectedCount} {selectedCount === 1 ? "item" : "items"} selected
      </span>
      <button
        onclick={handleClear}
        class="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
        title="Clear selection"
      >
        <IconClose class="w-4 h-4" />
      </button>
    </div>

    <!-- Common codes (applied to all) -->
    {#if commonCodes.length > 0}
      <div class="flex items-center gap-1">
        {#each commonCodes as code}
          <span
            class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs text-white"
            style:background-color={code.color}
          >
            {code.name}
            <button
              onclick={() => handleRemoveCode(code)}
              class="hover:bg-white/20 rounded-full p-0.5"
              title="Remove from all selected"
            >
              <IconClose class="w-3 h-3" />
            </button>
          </span>
        {/each}
      </div>
    {/if}

    <!-- Partial codes (applied to some) -->
    {#if partialCodes.length > 0}
      <div class="flex items-center gap-1">
        {#each partialCodes as code}
          <span
            class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs border-2 border-dashed"
            style:border-color={code.color}
            style:color={code.color}
            title="Applied to some selected items"
          >
            {code.name}
            <button
              onclick={() => handleApplyCode(code)}
              class="hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full p-0.5"
              title="Apply to all selected"
            >
              <IconPlus class="w-3 h-3" />
            </button>
          </span>
        {/each}
      </div>
    {/if}

    <!-- Apply code dropdown -->
    <PopupButton title="Apply Code" anchor="left">
      {#snippet button({ visible, toggle })}
        <button
          onclick={toggle}
          class="flex items-center gap-1 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          <IconPlus class="w-4 h-4" />
          Apply Code
        </button>
      {/snippet}

      <div class="w-64 max-h-80 overflow-y-auto">
        {#if availableCodes.length > 0}
          <div class="space-y-1 mb-3">
            {#each availableCodes as code}
              <button
                onclick={() => handleApplyCode(code)}
                class="w-full flex items-center gap-2 px-3 py-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-700 text-left"
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
        {:else if Object.keys(codingStore.codes).length === 0}
          <p class="text-sm text-slate-500 dark:text-slate-400 mb-3">
            No codes created yet
          </p>
        {:else}
          <p class="text-sm text-slate-500 dark:text-slate-400 mb-3">
            All codes already applied
          </p>
        {/if}

        <!-- Create new code -->
        <div class="border-t border-slate-200 dark:border-slate-700 pt-3">
          {#if showNewCodeInput}
            <div class="flex gap-2">
              <input
                type="text"
                bind:value={newCodeName}
                placeholder="New code name..."
                class="flex-1 px-2 py-1.5 text-sm border border-slate-300 dark:border-slate-600 rounded bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                onkeydown={(e) => {
                  if (e.key === "Enter") handleCreateAndApplyCode();
                  if (e.key === "Escape") showNewCodeInput = false;
                }}
              />
              <button
                onclick={handleCreateAndApplyCode}
                disabled={!newCodeName.trim()}
                class="px-2 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                Add
              </button>
            </div>
          {:else}
            <button
              onclick={() => (showNewCodeInput = true)}
              class="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700"
            >
              <IconPlus class="w-4 h-4" />
              Create new code
            </button>
          {/if}
        </div>
      </div>
    </PopupButton>

    <!-- View details button -->
    {#if onViewDetails && selectedCount === 1}
      <button
        onclick={onViewDetails}
        class="px-3 py-1.5 text-sm bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-md hover:bg-slate-200 dark:hover:bg-slate-700"
      >
        View Details
      </button>
    {/if}
  </div>
{/if}
