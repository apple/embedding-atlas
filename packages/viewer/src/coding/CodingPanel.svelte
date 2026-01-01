<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconChevronDown, IconChevronUp, IconClose, IconEdit, IconPlus } from "../assets/icons.js";
  import Button from "../widgets/Button.svelte";
  import Input from "../widgets/Input.svelte";
  import PopupButton from "../widgets/PopupButton.svelte";
  import Select from "../widgets/Select.svelte";
  import type { CodingStore } from "./store.svelte.js";
  import { codeColors, type Code } from "./types.js";

  interface Props {
    codingStore: CodingStore;
    onCodeClick?: (code: Code) => void;
    onExport?: () => void;
    onImport?: () => void;
  }

  let { codingStore, onCodeClick, onExport, onImport }: Props = $props();

  // New code form state
  let newCodeName = $state("");
  let newCodeDescription = $state("");
  let newCodeColor = $state(codeColors[0]);
  let newCodeParentId = $state<string | null>(null);
  let newCodeLevel = $state<1 | 2 | 3>(1);
  let showNewCodeForm = $state(false);

  // Edit code state
  let editingCode = $state<Code | null>(null);
  let editName = $state("");
  let editDescription = $state("");
  let editColor = $state("");

  // Collapse state for hierarchical codes
  let collapsedCodes = $state<Set<string>>(new Set());

  // Sort codes hierarchically
  let organizedCodes = $derived.by(() => {
    const rootCodes = codingStore.getRootCodes();
    const result: { code: Code; depth: number }[] = [];

    function addCodeWithChildren(code: Code, depth: number) {
      result.push({ code, depth });
      if (!collapsedCodes.has(code.id)) {
        const children = codingStore.getChildCodes(code.id);
        for (const child of children) {
          addCodeWithChildren(child, depth + 1);
        }
      }
    }

    for (const code of rootCodes.sort((a, b) => a.name.localeCompare(b.name))) {
      addCodeWithChildren(code, 0);
    }

    return result;
  });

  function handleCreateCode() {
    if (newCodeName.trim()) {
      codingStore.createCode(newCodeName.trim(), {
        description: newCodeDescription.trim() || undefined,
        color: newCodeColor,
        parentId: newCodeParentId,
        level: newCodeLevel,
      });
      resetNewCodeForm();
    }
  }

  function resetNewCodeForm() {
    newCodeName = "";
    newCodeDescription = "";
    newCodeColor = codeColors[Math.floor(Math.random() * codeColors.length)];
    newCodeParentId = null;
    newCodeLevel = 1;
    showNewCodeForm = false;
  }

  function startEdit(code: Code) {
    editingCode = code;
    editName = code.name;
    editDescription = code.description ?? "";
    editColor = code.color;
  }

  function saveEdit() {
    if (editingCode && editName.trim()) {
      codingStore.updateCode(editingCode.id, {
        name: editName.trim(),
        description: editDescription.trim() || undefined,
        color: editColor,
      });
      editingCode = null;
    }
  }

  function cancelEdit() {
    editingCode = null;
  }

  function toggleCollapse(codeId: string) {
    const newSet = new Set(collapsedCodes);
    if (newSet.has(codeId)) {
      newSet.delete(codeId);
    } else {
      newSet.add(codeId);
    }
    collapsedCodes = newSet;
  }

  function hasChildren(codeId: string): boolean {
    return codingStore.getChildCodes(codeId).length > 0;
  }

  let levelOptions = [
    { value: 1, label: "Level 1 - Open Coding" },
    { value: 2, label: "Level 2 - Axial Coding" },
    { value: 3, label: "Level 3 - Selective Coding" },
  ];

  let parentOptions = $derived([
    { value: null as any, label: "(No parent)" },
    ...Object.values(codingStore.codes).map((c) => ({
      value: c.id,
      label: c.name,
    })),
  ]);
</script>

<div class="flex flex-col h-full bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-700">
  <!-- Header -->
  <div class="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-700">
    <h2 class="text-lg font-semibold text-slate-800 dark:text-slate-200">
      Codes
    </h2>
    <div class="flex gap-2">
      <PopupButton title="Options" anchor="right">
        {#snippet button({ toggle })}
          <button
            onclick={toggle}
            class="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500"
            title="Options"
          >
            <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
            </svg>
          </button>
        {/snippet}

        <div class="flex flex-col gap-2 min-w-[160px]">
          {#if onExport}
            <button
              onclick={onExport}
              class="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-800 rounded"
            >
              Export Codes
            </button>
          {/if}
          {#if onImport}
            <button
              onclick={onImport}
              class="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-800 rounded"
            >
              Import Codes
            </button>
          {/if}
        </div>
      </PopupButton>
    </div>
  </div>

  <!-- Code list -->
  <div class="flex-1 overflow-y-auto">
    {#if organizedCodes.length === 0}
      <div class="p-4 text-center text-slate-500 dark:text-slate-400">
        <p class="mb-2">No codes created yet</p>
        <p class="text-sm">Click "New Code" below to start coding</p>
      </div>
    {:else}
      <div class="py-2">
        {#each organizedCodes as { code, depth }}
          <div
            class="group flex items-center gap-2 px-3 py-2 hover:bg-slate-50 dark:hover:bg-slate-800 cursor-pointer"
            style:padding-left="{12 + depth * 16}px"
            role="button"
            tabindex="0"
            onclick={() => onCodeClick?.(code)}
            onkeydown={(e) => e.key === "Enter" && onCodeClick?.(code)}
          >
            <!-- Collapse toggle -->
            {#if hasChildren(code.id)}
              <button
                onclick={(e) => {
                  e.stopPropagation();
                  toggleCollapse(code.id);
                }}
                class="p-0.5 rounded hover:bg-slate-200 dark:hover:bg-slate-700"
              >
                {#if collapsedCodes.has(code.id)}
                  <IconChevronDown class="w-4 h-4 text-slate-400" />
                {:else}
                  <IconChevronUp class="w-4 h-4 text-slate-400" />
                {/if}
              </button>
            {:else}
              <span class="w-5"></span>
            {/if}

            <!-- Color indicator -->
            <span
              class="w-3 h-3 rounded-full flex-shrink-0"
              style:background-color={code.color}
            ></span>

            <!-- Code name -->
            {#if editingCode?.id === code.id}
              <input
                type="text"
                bind:value={editName}
                class="flex-1 px-2 py-1 text-sm border border-slate-300 dark:border-slate-600 rounded bg-white dark:bg-slate-800"
                onclick={(e) => e.stopPropagation()}
                onkeydown={(e) => {
                  if (e.key === "Enter") saveEdit();
                  if (e.key === "Escape") cancelEdit();
                }}
              />
              <button
                onclick={(e) => {
                  e.stopPropagation();
                  saveEdit();
                }}
                class="p-1 text-green-600 hover:bg-green-50 dark:hover:bg-green-900/20 rounded"
              >
                ✓
              </button>
              <button
                onclick={(e) => {
                  e.stopPropagation();
                  cancelEdit();
                }}
                class="p-1 text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 rounded"
              >
                <IconClose class="w-4 h-4" />
              </button>
            {:else}
              <span class="flex-1 text-sm text-slate-700 dark:text-slate-300 truncate">
                {code.name}
              </span>

              <!-- Frequency badge -->
              {#if code.frequency > 0}
                <span class="text-xs text-slate-400 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded-full">
                  {code.frequency}
                </span>
              {/if}

              <!-- Level indicator -->
              <span
                class="text-xs px-1.5 py-0.5 rounded {
                  code.level === 1 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                  code.level === 2 ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' :
                  'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
                }"
              >
                L{code.level}
              </span>

              <!-- Edit button -->
              <button
                onclick={(e) => {
                  e.stopPropagation();
                  startEdit(code);
                }}
                class="p-1 opacity-0 group-hover:opacity-100 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 rounded transition-opacity"
                title="Edit code"
              >
                <IconEdit class="w-4 h-4" />
              </button>

              <!-- Delete button -->
              <button
                onclick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Delete code "${code.name}"?`)) {
                    codingStore.deleteCode(code.id);
                  }
                }}
                class="p-1 opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-opacity"
                title="Delete code"
              >
                <IconClose class="w-4 h-4" />
              </button>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- New code form -->
  <div class="border-t border-slate-200 dark:border-slate-700 p-4">
    {#if showNewCodeForm}
      <div class="space-y-3">
        <div>
          <label class="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
            Code Name
          </label>
          <Input
            type="text"
            bind:value={newCodeName}
            placeholder="Enter code name..."
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
            Description (optional)
          </label>
          <textarea
            bind:value={newCodeDescription}
            placeholder="Describe what this code represents..."
            class="w-full px-3 py-2 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            rows="2"
          ></textarea>
        </div>

        <div class="flex gap-3">
          <div class="flex-1">
            <label class="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
              Color
            </label>
            <div class="flex gap-1 flex-wrap">
              {#each codeColors as color}
                <button
                  onclick={() => (newCodeColor = color)}
                  class="w-6 h-6 rounded-full border-2 transition-transform hover:scale-110"
                  style:background-color={color}
                  class:border-slate-400={newCodeColor !== color}
                  class:border-slate-800={newCodeColor === color}
                  class:dark:border-white={newCodeColor === color}
                  class:scale-110={newCodeColor === color}
                ></button>
              {/each}
            </div>
          </div>
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
            Level
          </label>
          <Select
            options={levelOptions}
            value={newCodeLevel}
            onChange={(v) => (newCodeLevel = v)}
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">
            Parent Code
          </label>
          <Select
            options={parentOptions}
            value={newCodeParentId}
            onChange={(v) => (newCodeParentId = v)}
          />
        </div>

        <div class="flex gap-2 pt-2">
          <Button
            label="Create Code"
            onClick={handleCreateCode}
            class="flex-1"
          />
          <Button
            label="Cancel"
            onClick={resetNewCodeForm}
            class="flex-1"
          />
        </div>
      </div>
    {:else}
      <button
        onclick={() => (showNewCodeForm = true)}
        class="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
      >
        <IconPlus class="w-4 h-4" />
        New Code
      </button>
    {/if}
  </div>

  <!-- Statistics -->
  <div class="px-4 py-3 bg-slate-50 dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
    <div class="flex justify-between text-xs text-slate-500 dark:text-slate-400">
      <span>{Object.keys(codingStore.codes).length} codes</span>
      <span>{Object.keys(codingStore.codeApplications).length} coded items</span>
    </div>
  </div>
</div>
