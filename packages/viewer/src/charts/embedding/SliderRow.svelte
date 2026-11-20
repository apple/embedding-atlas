<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Slider from "../../widgets/Slider.svelte";

  interface Props {
    label: string;
    value: number;
    min: number;
    max: number;
    step?: number;
    title?: string;
    /** An optional toggle button after the slider (e.g., "Auto"). When `active`, the slider shows the automatic value. */
    toggle?: { label: string; active: boolean; title?: string; onClick: () => void } | null;
    onChange: (value: number) => void;
  }

  let { label, value, min, max, step, title, toggle = null, onChange }: Props = $props();
</script>

<div class="flex gap-2 items-center" title={title}>
  <div class="w-20 flex-none text-sm text-slate-500 dark:text-slate-400 select-none">{label}</div>
  <div class="flex-1 min-w-0" class:opacity-60={toggle?.active}>
    <Slider fill bind:value={() => value, onChange} min={min} max={max} step={step} />
  </div>
  {#if toggle != null}
    <button
      class="flex-none px-1.5 rounded-md text-xs leading-5 select-none border focus-visible:outline-2 outline-blue-600 -outline-offset-1 {toggle.active
        ? 'bg-blue-500 border-blue-500 text-white'
        : 'bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-500 dark:text-slate-400'}"
      title={toggle.title ?? toggle.label}
      onclick={toggle.onClick}
    >
      {toggle.label}
    </button>
  {/if}
</div>
