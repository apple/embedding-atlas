<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";

  import { IconDown, IconRight } from "../assets/icons.js";

  interface Props {
    label: string;
    class?: string | null;
    children?: Snippet;
  }

  let { label, class: className = null, children }: Props = $props();

  let open = $state(false);
</script>

<button
  class="flex items-center gap-1 text-sm text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 select-none {className ??
    ''}"
  onclick={() => (open = !open)}
>
  {#if open}
    <IconDown class="w-3.5 h-3.5" />
  {:else}
    <IconRight class="w-3.5 h-3.5" />
  {/if}
  {label}
</button>
{#if open}
  {@render children?.()}
{/if}
