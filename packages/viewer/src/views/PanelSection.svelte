<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    title: string;
    children: Snippet;
  }

  let { title, children }: Props = $props();
</script>

<div
  class="flex flex-col gap-2 bg-white dark:bg-slate-800 p-2 rounded-md border border-slate-200 dark:border-slate-700 select-none"
>
  <div class="text-slate-700 dark:text-slate-300 font-medium">
    {title}
  </div>
  {@render children()}
</div>
