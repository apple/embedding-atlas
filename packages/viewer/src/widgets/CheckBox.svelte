<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconCheckboxChecked, IconCheckboxUnchecked } from "../assets/icons.js";

  interface Props {
    checked: boolean;
    label?: string | null;
    disabled?: boolean;
    class?: string;
  }

  let { checked = $bindable(), label = null, disabled = false, class: className = "" }: Props = $props();
</script>

<label class="flex py-1 my-[1px] gap-1 items-center select-none {disabled ? 'pointer-events-none' : ''} {className}">
  <input type="checkbox" bind:checked={checked} disabled={disabled} class="sr-only" />
  {#if checked}
    <IconCheckboxChecked class="w-6 h-6 {disabled ? 'text-slate-400 dark:text-slate-500' : 'text-blue-500'}" />
  {:else}
    <IconCheckboxUnchecked class="w-6 h-6 text-slate-400 dark:text-slate-500" />
  {/if}
  {#if label != null}
    <span class="text-[13px] {disabled ? 'text-slate-400 dark:text-slate-500' : 'dark:text-slate-300'}">{label}</span>
  {/if}
</label>
