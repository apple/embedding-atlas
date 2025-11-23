<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    label?: string | null;
    icon?: any | null;
    title?: string;
    order?: number | null;
    class?: string | null;
    onClick?: () => void;
  }

  let { label = null, icon = null, title = "", order = null, onClick, class: className }: Props = $props();
</script>

<button
  class="rounded-md flex select-none items-center px-1.5 py-1.5 h-[28px] border bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 dark:text-slate-400 focus-visible:outline-2 outline-blue-600 -outline-offset-1 {className ??
    ''}"
  style:order={order}
  title={title}
  onclick={() => {
    onClick?.();
  }}
>
  {#if icon != null}
    {@const SvelteComponent = icon}
    <SvelteComponent class="w-5 h-5" />
  {/if}
  {#if label != null && label != ""}
    <span class="mx-1">{label}</span>
  {/if}
</button>
