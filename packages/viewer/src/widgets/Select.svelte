<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import SelectBase, { type Props as BaseProps } from "./SelectBase.svelte";

  type Props = BaseProps & {
    label?: string | null | undefined;
  };

  let props: Props = $props();

  let className =
    "form-select rounded-md py-1 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 dark:text-slate-400 select-none text-ellipsis";
</script>

{#if props.label != null}
  <label class="select-none flex items-center gap-2">
    <span class="text-slate-500 dark:text-slate-400 whitespace-nowrap">{props.label}</span>
    <SelectBase {...props} class={className + " " + (props.class ?? "")} />
  </label>
{:else}
  <SelectBase {...props} class={className + " " + (props.class ?? "")} />
{/if}
