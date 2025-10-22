<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  export interface Message {
    text: string;
    progress?: number;
    error?: boolean;
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
  import Spinner from "../widgets/Spinner.svelte";

  interface Props {
    messages?: Message[];
  }
  let { messages = [] }: Props = $props();
</script>

<div
  class="flex flex-col p-4 w-[420px] border rounded-md bg-slate-50 border-slate-300 dark:bg-slate-900 dark:border-slate-700"
>
  {#each messages as m, i}
    {@const isLast = i == messages.length - 1}
    <div class="flex items-center text-slate-500 dark:text-slate-500">
      <div class="w-7 flex-none">
        {#if isLast}
          <Spinner status={null} />
        {/if}
      </div>
      <div class="flex-1" class:text-red-400={m.error}>
        {m.text}
      </div>
      <div class="flex-none">
        {#if m.progress != null && isLast}
          {m.progress.toFixed(0)}%
        {/if}
      </div>
    </div>
  {/each}
</div>
