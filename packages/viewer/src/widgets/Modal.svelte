<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { type Snippet } from "svelte";
  import { fade, scale } from "svelte/transition";

  import { IconClose } from "../assets/icons.js";

  interface Props {
    open: boolean;
    title?: string;
    size?: "sm" | "md" | "lg" | "xl" | "full";
    onClose: () => void;
    header?: Snippet;
    footer?: Snippet;
    children?: Snippet;
  }

  let { open, title = "", size = "md", onClose, header, footer, children }: Props = $props();

  function onKeyDown(e: KeyboardEvent) {
    if (open && e.key === "Escape") {
      onClose();
      e.stopPropagation();
    }
  }

  function onBackdropClick(e: MouseEvent) {
    if (e.target === e.currentTarget) {
      onClose();
    }
  }

  const sizeClasses: Record<string, string> = {
    sm: "max-w-sm",
    md: "max-w-lg",
    lg: "max-w-2xl",
    xl: "max-w-4xl",
    full: "max-w-[90vw] max-h-[90vh]",
  };
</script>

<svelte:window onkeydown={onKeyDown} />

{#if open}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
    transition:fade={{ duration: 150 }}
    onclick={onBackdropClick}
  >
    <div
      class="relative flex flex-col bg-white dark:bg-slate-900 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 overflow-hidden {sizeClasses[size]}"
      style:max-height="90vh"
      transition:scale={{ duration: 150, start: 0.95 }}
    >
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
        {#if header}
          {@render header()}
        {:else if title}
          <h2 class="text-lg font-semibold text-slate-800 dark:text-slate-200">{title}</h2>
        {:else}
          <div></div>
        {/if}
        <button
          onclick={onClose}
          class="p-1.5 rounded-md text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          title="Close"
        >
          <IconClose class="w-5 h-5" />
        </button>
      </div>

      <!-- Body -->
      <div class="flex-1 overflow-auto p-4">
        {@render children?.()}
      </div>

      <!-- Footer -->
      {#if footer}
        <div class="px-4 py-3 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
          {@render footer()}
        </div>
      {/if}
    </div>
  </div>
{/if}
