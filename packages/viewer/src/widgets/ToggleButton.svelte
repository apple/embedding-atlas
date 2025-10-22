<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    title?: string;
    checked: boolean;
    label?: string | null;
    icon?: any | null;
  }

  let { title = "", checked = $bindable(), label = null, icon = null }: Props = $props();
</script>

<button
  class="rounded-md px-1.5 py-1.5 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 flex select-none items-center focus-visible:outline-2 outline-blue-600 -outline-offset-1"
  class:text-slate-400={!checked}
  class:dark:text-slate-500={!checked}
  title={title}
  onclick={() => {
    checked = !checked;
  }}
>
  {#if icon != null}
    {@const SvelteComponent = icon}
    <SvelteComponent class="w-5 h-5" />
  {/if}
  {#if label != null}
    <span class="flex h-5 items-center">{label}</span>
  {/if}
</button>
