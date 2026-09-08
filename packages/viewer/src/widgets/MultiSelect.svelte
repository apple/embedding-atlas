<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  export interface Option {
    label: string;
    value: any;
  }

  export interface Props {
    label?: string | null;
    options?: Option[];
    values: any[];
    placeholder?: string | null;
    onChange?: (values: any[]) => void;
    class?: string | null;
    title?: string;
  }
</script>

<script lang="ts">
  import { IconCheckboxChecked, IconCheckboxUnchecked, IconChevronDown } from "../assets/icons.js";

  let {
    label = null,
    options = [],
    values,
    placeholder = "Select…",
    onChange = undefined,
    class: className = "",
    title = "",
  }: Props = $props();

  let container: HTMLDivElement;
  let isOpen = $state(false);

  // JSON-stringify keys to compare deeply.
  let selectedSet = $derived(new Set(values.map((v) => JSON.stringify(v))));
  let matched = $derived(options.filter((o) => selectedSet.has(JSON.stringify(o.value))));
  let allChecked = $derived(options.length > 0 && matched.length === options.length);

  let summary = $derived.by(() => {
    if (options.length === 0) return placeholder ?? "";
    if (allChecked) return "All";
    if (matched.length === 0) return placeholder ?? "";
    if (matched.length <= 3) return matched.map((x) => x.label).join(", ");
    return `${matched.length} selected`;
  });

  function toggle(value: any) {
    const key = JSON.stringify(value);
    const next = selectedSet.has(key) ? values.filter((v) => JSON.stringify(v) !== key) : [...values, value];
    onChange?.(next);
  }

  function toggleAll() {
    onChange?.(allChecked ? [] : options.map((o) => o.value));
  }

  function onFocusOut(e: FocusEvent) {
    if (e.relatedTarget && e.relatedTarget instanceof Node && container.contains(e.relatedTarget)) {
      return;
    }
    isOpen = false;
  }

  function onKeyDown(e: KeyboardEvent) {
    if (isOpen && e.key === "Escape") {
      isOpen = false;
      e.stopPropagation();
    }
  }
</script>

{#snippet dropdown()}
  {#if isOpen}
    <ul
      class="absolute mt-1 rounded-md shadow-md p-1 z-20 flex flex-col bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 min-w-full max-w-96 max-h-[320px] overflow-y-auto"
    >
      <button
        type="button"
        class="flex px-2 py-1 gap-1 items-center select-none text-left rounded-md text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 w-full"
        onclick={toggleAll}
      >
        {#if allChecked}
          <IconCheckboxChecked class="w-6 h-6 text-blue-500 flex-none" />
        {:else}
          <IconCheckboxUnchecked class="w-6 h-6 text-slate-400 dark:text-slate-500 flex-none" />
        {/if}
        <span class="text-[13px] truncate">Select All</span>
      </button>
      {#if options.length > 0}
        <hr class="border-slate-200 dark:border-slate-600 my-1" />
      {/if}
      {#each options as opt (JSON.stringify(opt.value))}
        <button
          type="button"
          class="flex px-2 py-1 gap-1 items-center select-none text-left rounded-md text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 w-full"
          onclick={() => toggle(opt.value)}
        >
          {#if selectedSet.has(JSON.stringify(opt.value))}
            <IconCheckboxChecked class="w-6 h-6 text-blue-500 flex-none" />
          {:else}
            <IconCheckboxUnchecked class="w-6 h-6 text-slate-400 dark:text-slate-500 flex-none" />
          {/if}
          <span class="text-[13px] truncate">{opt.label}</span>
        </button>
      {/each}
    </ul>
  {/if}
{/snippet}

{#snippet trigger(triggerClass: string)}
  <button
    type="button"
    class="rounded-md h-[28px] text-[13px] text-left pl-3 pr-2 bg-white dark:bg-slate-900
      border border-slate-300 dark:border-slate-600 dark:text-slate-400 select-none
      flex items-center gap-2 focus-visible:outline-2 outline-blue-600 -outline-offset-1
      {triggerClass}"
    onclick={() => (isOpen = !isOpen)}
    title={title}
  >
    <span class="overflow-hidden whitespace-nowrap text-ellipsis flex-1">{summary}</span>
    <IconChevronDown class="w-4 h-4 flex-none" />
  </button>
{/snippet}

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  bind:this={container}
  class={label != null ? `select-none flex items-center gap-2 ${className ?? ""}` : "relative"}
  onfocusout={onFocusOut}
  onkeydown={onKeyDown}
>
  {#if label != null}
    <span class="text-slate-500 dark:text-slate-400 whitespace-nowrap text-sm">{label}</span>
    <div class="relative">
      {@render trigger("")}
      {@render dropdown()}
    </div>
  {:else}
    {@render trigger(className ?? "")}
    {@render dropdown()}
  {/if}
</div>
