<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    label?: string | null;
    icon?: any | null;
    title?: string;
    order?: number | null;
    class?: string | null;
    disabled?: boolean;
    onClick?: () => void;
  }

  let {
    label = null,
    icon = null,
    title = "",
    order = null,
    onClick,
    class: className,
    disabled = false,
  }: Props = $props();
</script>

<button
  class="rounded-md flex select-none items-center px-1.5 py-1.5 h-[28px] w-[28px]
    text-slate-500 disabled:text-slate-400/60 dark:text-slate-400 dark:disabled:text-slate-600
    hover:bg-slate-100 dark:hover:bg-slate-700
    focus-visible:outline-2 outline-blue-600 -outline-offset-1
    disabled:pointer-events-none
    {className ?? ''}"
  style:order={order}
  title={title}
  disabled={disabled}
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
