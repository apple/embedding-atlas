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
  class="rounded-md flex select-none items-center px-1 py-1 text-slate-500 bg-slate-100 dark:text-slate-400 dark:bg-slate-800 focus-visible:outline-2 outline-blue-600 -outline-offset-1 {className ??
    ''}"
  style:order={order}
  title={title}
  onclick={() => {
    onClick?.();
  }}
>
  {#if icon != null}
    {@const SvelteComponent = icon}
    <SvelteComponent class="w-4 h-4" />
  {/if}
  {#if label != null && label != ""}
    <span class="mx-1">{label}</span>
  {/if}
</button>
