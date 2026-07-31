<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    title: string;
    /** Optional buttons rendered at the top-right corner, inline with the title. */
    actions?: Snippet;
    children: Snippet;
  }

  let { title, actions, children }: Props = $props();
</script>

<div
  class="flex flex-col gap-2 bg-white dark:bg-slate-800 p-2 rounded-md border border-slate-200 dark:border-slate-700 select-none"
>
  <div class="flex items-center justify-between gap-2 min-h-[20px]">
    <div class="text-slate-700 dark:text-slate-300 font-medium">
      {title}
    </div>
    {#if actions}
      <div class="flex items-center gap-1">
        {@render actions()}
      </div>
    {/if}
  </div>
  {@render children()}
</div>
