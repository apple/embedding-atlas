<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { AdditionalHeaderContent, AdditionalHeaderContentProps } from "../../api/custom-headers";
  import { Context } from "../../context/context.svelte";

  interface Props {
    col: string;
    customHeader: AdditionalHeaderContent;
  }

  let { col, customHeader }: Props = $props();

  const model = Context.model;

  const customCellAction = (component: AdditionalHeaderContent) => {
    if (typeof component == "function") {
      return (node: HTMLDivElement, props: AdditionalHeaderContentProps) => {
        let obj = new component(node, props);
        return {
          ...(obj.update ? { update: obj.update.bind(obj) } : {}),
          ...(obj.destroy ? { destroy: obj.destroy.bind(obj) } : {}),
        };
      };
    } else {
      return (node: HTMLDivElement, props: AdditionalHeaderContentProps) => {
        let obj = new component.class(node, props);
        return {
          ...(obj.update ? { update: obj.update.bind(obj) } : {}),
          ...(obj.destroy ? { destroy: obj.destroy.bind(obj) } : {}),
        };
      };
    }
  };

  let action = $derived(customCellAction(customHeader));
</script>

<div use:action={{ column: col }}></div>

<style>
</style>
