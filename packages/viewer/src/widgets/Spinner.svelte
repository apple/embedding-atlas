<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import SvgSpinner from "~icons/svg-spinners/270-ring-with-bg";

  interface Props {
    status?: string | null;
  }

  let { status = "Loading..." }: Props = $props();
</script>

<div role="status" class="flex flex-row items-center">
  <SvgSpinner class="text-blue-500" />
  {#if status != null}
    <span class="pl-2 text-slate-500 dark:text-slate-500">{status}</span>
  {/if}
</div>
