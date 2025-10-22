<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import InlineSelect from "../../widgets/InlineSelect.svelte";

  import type { ScaleType } from "../common/types.js";

  interface Props {
    label: string;
    value: ScaleType | null;
  }
  let { label, value = $bindable() }: Props = $props();

  const candidates: ScaleType[] = ["linear", "log", "symlog"];
</script>

{#if value != null && value != "band"}
  <span class="flex gap-1 select-none">
    <span class="text-slate-400 dark:text-slate-500 text-sm">{label}:</span>
    <InlineSelect
      options={candidates.map((c) => ({ value: c, label: c }))}
      value={value}
      onChange={(v) => (value = v)}
    />
  </span>
{/if}
