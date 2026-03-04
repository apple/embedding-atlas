<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import TooltipContent from "../../views/TooltipContent.svelte";

  import type { ColumnStyle } from "../../renderers/types.js";
  import { compileLiquidTemplate } from "../../utils/html_template.js";
  import type { RowID } from "../chart.js";

  interface Props {
    data: Record<string, any>[];
    columns: string[];
    columnStyles: Record<string, ColumnStyle>;
    highlight: RowID[] | null;
    cardTemplate?: string;
    onRowClick: (rowId: RowID | null | undefined, event: MouseEvent) => void;
  }

  let { data, columns, columnStyles, highlight, cardTemplate, onRowClick }: Props = $props();

  let highlightSet = $derived(new Set(highlight));
  let idMapper = new Map<RowID, Element>();

  let compiledTemplate = $derived(cardTemplate != null ? compileLiquidTemplate(cardTemplate) : undefined);

  export function getElementForId(id: RowID): Element | undefined {
    return idMapper.get(id);
  }
</script>

<div class="p-2">
  <div class="grid gap-2" style:grid-template-columns="repeat(auto-fill, minmax(300px, 1fr))">
    {#each data as row}
      {@const rowId = row.__id__}
      {@const values = Object.fromEntries(columns.map((col) => [col, row[col]]))}
      {@const highlighted = highlightSet.has(rowId)}
      <div>
        <button
          bind:this={() => idMapper.get(rowId), (v) => idMapper.set(rowId, v)}
          class="flex items-stretch flex-col border rounded-lg transition-all hover:shadow-md bg-white dark:bg-slate-800 overflow-hidden text-left w-full select-text appearance-none"
          class:border-slate-200={!highlighted}
          class:dark:border-slate-700={!highlighted}
          class:border-blue-500={highlighted}
          class:ring-1={highlighted}
          class:ring-blue-500={highlighted}
          onmousedown={(e) => {
            // Prevent text selection when shift-click to multi-select
            if (e.shiftKey || e.ctrlKey || e.metaKey) {
              e.preventDefault();
            }
          }}
          onclick={(e) => onRowClick(rowId, e)}
        >
          {#if compiledTemplate != null}
            {@html compiledTemplate(values)}
          {:else}
            <div class="w-full p-4">
              <TooltipContent values={values} columns={columns} columnStyles={columnStyles} />
            </div>
          {/if}
        </button>
      </div>
    {/each}
  </div>
</div>
