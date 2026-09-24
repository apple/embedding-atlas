<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  export interface Option {
    label: string;
    value: any;
  }

  export interface Props {
    options?: Option[];
    /** The currently selected values, in order. */
    values: any[];
    /** Placeholder for the "add" row. */
    placeholder?: string | null;
    onChange?: (values: any[]) => void;
    class?: string | null;
  }
</script>

<script lang="ts">
  import { IconClose, IconDown, IconUp } from "../assets/icons.js";
  import Select from "./Select.svelte";

  let { options = [], values, placeholder = "(add)", onChange = undefined, class: className = "" }: Props = $props();

  // JSON-stringify keys to compare deeply (consistent with MultiSelect).
  let key = (v: any) => JSON.stringify(v);
  let selectedSet = $derived(new Set(values.map(key)));
  // Options not yet selected, used for the "add" row.
  let availableOptions = $derived(options.filter((o) => !selectedSet.has(key(o.value))));

  // Options offered for the row at index `i`: everything not selected elsewhere,
  // plus the row's own current value (so it can be changed or kept).
  function optionsForRow(i: number): Option[] {
    return options.filter((o) => !selectedSet.has(key(o.value)) || key(o.value) === key(values[i]));
  }

  function replaceAt(i: number, value: any) {
    let next = [...values];
    next[i] = value;
    onChange?.(next);
  }

  function removeAt(i: number) {
    onChange?.(values.filter((_, index) => index !== i));
  }

  function swap(i: number, j: number) {
    if (j < 0 || j >= values.length) {
      return;
    }
    let next = [...values];
    [next[i], next[j]] = [next[j], next[i]];
    onChange?.(next);
  }

  function add(value: any) {
    if (value === undefined) {
      return;
    }
    onChange?.([...values, value]);
  }
</script>

<div class={`flex flex-col gap-1 ${className ?? ""}`}>
  {#each values as value, i (i)}
    <div class="flex items-center gap-1">
      <Select
        value={value}
        onChange={(v) => replaceAt(i, v)}
        class="w-full min-w-0 flex-1"
        options={optionsForRow(i)}
      />
      <button
        type="button"
        class="rounded-md p-1 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-30"
        title="Move up"
        disabled={i === 0}
        onclick={() => swap(i, i - 1)}
      >
        <IconUp class="w-4 h-4" />
      </button>
      <button
        type="button"
        class="rounded-md p-1 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-30"
        title="Move down"
        disabled={i === values.length - 1}
        onclick={() => swap(i, i + 1)}
      >
        <IconDown class="w-4 h-4" />
      </button>
      <button
        type="button"
        class="rounded-md p-1 text-slate-500 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700"
        title="Remove"
        onclick={() => removeAt(i)}
      >
        <IconClose class="w-4 h-4" />
      </button>
    </div>
  {/each}
  {#if availableOptions.length > 0}
    <Select value={undefined} onChange={add} placeholder={placeholder} class="w-full" options={availableOptions} />
  {/if}
</div>
