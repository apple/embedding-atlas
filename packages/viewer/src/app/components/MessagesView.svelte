<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  export interface Message {
    text: string;
    progress?: number;
    progressText?: string;
    error?: boolean;
    markdown?: boolean;
  }

  export function appendedMessages(target: Message[], message: Message): Message[] {
    if (target.length > 0 && target[target.length - 1].text == message.text) {
      let r = target.slice();
      r[r.length - 1] = message;
      return r;
    } else {
      return [...target, message];
    }
  }
</script>

<script lang="ts">
  import { IconError, IconSpinner } from "../../assets/icons.js";
  import { marked } from "marked";
  import DOMPurify from "dompurify";

  interface Props {
    messages?: Message[];
  }
  let { messages = [] }: Props = $props();

  function renderText(m: Message): string {
    if (m.markdown) {
      const rawHtml = marked.parse(m.text, { async: false }) as string;
      return DOMPurify.sanitize(rawHtml);
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
        ? 'text-slate-700 dark:text-slate-300'
        : 'text-slate-500 dark:text-slate-400'}"
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
