<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { IconError, IconSpinner } from "../../assets/icons.js";
  import { renderMarkdown } from "../../renderers/markdown.js";
  import type { LogMessage } from "../logging.js";

  interface Props {
    messages?: LogMessage[];
  }
  let { messages = [] }: Props = $props();

  function renderText(m: LogMessage): string {
    if (m.markdown) {
      return renderMarkdown(m.text);
    }
    return m.text;
  }
</script>

<div
  class="flex flex-col gap-1 p-4 w-[420px] border rounded-md bg-slate-50 border-slate-300 dark:bg-slate-900 dark:border-slate-700 select-text"
>
  {#each messages as m, i}
    {@const isLast = i == messages.length - 1}
    <div
      class="flex items-start leading-5 {isLast
        ? 'text-slate-500 dark:text-slate-400'
        : 'text-slate-300 dark:text-slate-600'}"
    >
      <div class="w-7 flex-none">
        {#if isLast || m.error}
          {#if m.error}
            <IconError class="text-red-400 w-5 h-5" />
          {:else}
            <IconSpinner class="text-blue-500 w-5 h-5" />
          {/if}
        {/if}
      </div>
      <div class="flex-1" class:text-red-400={m.error}>
        {#if m.markdown}
          <div class="markdown-content">
            {@html renderText(m)}
          </div>
        {:else}
          {m.text}
        {/if}
      </div>
      {#if isLast}
        <div class="flex-none font-mono text-sm">
          {#if m.progress != null}
            {m.progress.toFixed(0)}%
          {/if}
          {#if m.progressText != null}
            {m.progressText}
          {/if}
        </div>
      {/if}
    </div>
  {/each}
</div>

<style>
  :global(.markdown-content p) {
    display: inline;
  }

  :global(.markdown-content a) {
    text-decoration: underline;
  }
</style>
