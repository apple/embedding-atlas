<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { CustomCell, CustomCellProps } from "../../../api/custom-cells.js";
  import { Context } from "../../../context/context.svelte";

  interface Props {
    row: string;
    col: string;
    customCell: CustomCell;
    height: number;
  }

  let { row, col, customCell, height = $bindable() }: Props = $props();

  const model = Context.model;

  const customCellAction = (component: CustomCell) => {
    if (typeof component == "function") {
      return (node: HTMLDivElement, props: CustomCellProps) => {
        let obj = new component(node, props);
        return {
          ...(obj.update ? { update: obj.update.bind(obj) } : {}),
          ...(obj.destroy ? { destroy: obj.destroy.bind(obj) } : {}),
        };
      };
    } else {
      return (node: HTMLDivElement, props: CustomCellProps) => {
        let obj = new component.class(node, props);
        return {
          ...(obj.update ? { update: obj.update.bind(obj) } : {}),
          ...(obj.destroy ? { destroy: obj.destroy.bind(obj) } : {}),
        };
      };
    }
  };

  let action = $derived(customCellAction(customCell));
  let value = $derived(model.getContent({ row, col }));
  let rowData = $derived(model.getRowData(row));
</script>

<div use:action={{ value, rowData }} bind:clientHeight={height}></div>

<style>
</style>
