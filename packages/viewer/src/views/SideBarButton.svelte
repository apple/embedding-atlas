<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    icon: any;
    title?: string;

    active?: boolean;
    open?: boolean;

    onClick?: () => void;
  }

  let { icon, onClick, title, active, open }: Props = $props();

  let tooltipGate = $state(true);

  let Icon = $derived(icon);
</script>

<button
  class="group relative p-[calc(var(--spacing)*0.5+1px)] flex items-center justify-center rounded-md
    focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-slate-400 dark:focus-visible:ring-slate-500
    {active
    ? 'bg-slate-50 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
    : 'text-slate-400 hover:text-slate-500 dark:text-slate-500 dark:hover:text-slate-400'}"
  onclick={() => {
    tooltipGate = false;
    onClick?.();
  }}
  onmouseleave={() => (tooltipGate = true)}
  aria-label={title}
>
  <Icon class="w-7 h-7" />
  {#if title && tooltipGate}
    <span
      class="absolute left-[calc(100%+4px)] top-1/2 -translate-y-1/2 whitespace-nowrap
        px-2 py-1 rounded text-md leading-tight shadow-sm
        bg-slate-100 text-slate-500 border border-slate-300
        dark:bg-slate-800 dark:text-slate-400 dark:border-slate-700
        pointer-events-none z-50 opacity-0 group-hover:opacity-100 transition-opacity">{title}</span
    >
  {/if}
</button>
