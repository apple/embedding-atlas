<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import ColumnStylePickerRow from "./ColumnStylePickerRow.svelte";

  import { type ColumnStyle } from "../renderers/index.js";
  import type { ColumnDesc } from "../utils/database.js";

  interface Props {
    columns: ColumnDesc[];
    styles: Record<string, ColumnStyle>;
    onStylesChange: (value: Record<string, ColumnStyle>) => void;
  }

  let { columns, styles, onStylesChange }: Props = $props();
</script>

<div class="max-h-60 overflow-x-hidden overflow-y-scroll border border-slate-300 dark:border-slate-700 p-2 rounded-md">
  <table>
    <thead>
      <tr class="select-none">
        <th class="pb-2 text-slate-500 dark:text-slate-400 text-left font-normal text-sm">Column</th>
        <th class="pb-2 text-slate-500 dark:text-slate-400 text-left font-normal text-sm">Format</th>
        <th class="pb-2 text-slate-500 dark:text-slate-400 text-left font-normal text-sm">Style</th>
      </tr>
    </thead>
    <tbody>
      {#each columns as column}
        <ColumnStylePickerRow
          column={column}
          style={styles[column.name] ?? {}}
          onChange={(s) => {
            let newStyles = { ...styles };
            newStyles[column.name] = s;
            onStylesChange(newStyles);
          }}
        />
      {/each}
    </tbody>
  </table>
</div>
