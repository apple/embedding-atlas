<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    value: string;
    options: {
      value: string;
      icon?: any;
      label?: string;
      title?: string;
    }[];
    onChange?: (value: string) => void;
  }

  let { value, options, onChange }: Props = $props();
</script>

<div
  class="flex rounded-md select-none items-center overflow-hidden border border-slate-300 dark:border-slate-600 dark:text-slate-400"
>
  {#each options as opt}
    {@const isSelected = value == opt.value}
    <button
      class="flex items-center px-1.5 py-1.5 h-[26px] rounded-[3.875px] focus-visible:outline-2 outline-blue-600 -outline-offset-1 {isSelected
        ? 'bg-white dark:bg-black shadow-md'
        : 'text-slate-400 dark:text-slate-600'}"
      title={opt.title ?? opt.label}
      onclick={() => {
        onChange?.(opt.value);
      }}
    >
      {#if opt.icon != null}
        {@const SvelteComponent = opt.icon}
        <SvelteComponent class="w-5 h-5" />
      {/if}
      {#if opt.label != null && opt.label != ""}
        <span class="mx-1">{opt.label}</span>
      {/if}
    </button>
  {/each}
</div>
