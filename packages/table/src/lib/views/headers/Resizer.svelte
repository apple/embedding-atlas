<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { Context } from "../../context/context.svelte.js";
  import { ColResizeController } from "../../controllers/ColResizeController.svelte.js";

  interface Props {
    col: string;
  }

  let { col }: Props = $props();

  const model = Context.model;

  let controller = new ColResizeController({
    tableModel: model,
    tableController: Context.controller,
    col,
  });

  const x = $derived(model.colPositions[col] + model.colWidths[col]);
</script>

<div
  class="header-resize-indicator"
  style:--x={x + "px"}
  onpointerdown={controller.handlePointerDown}
  onpointermove={controller.handlePointerMove}
  onpointerup={controller.handlePointerUp}
>
  <div class="pill"></div>
</div>

<style>
  .header-resize-indicator {
    position: absolute;
    z-index: 2;
    box-sizing: border-box;
    width: 12px;
    height: calc(100% - 6px);
    margin: 2px;
    cursor: col-resize;
    justify-content: center;
    display: flex;
    align-items: center;
    justify-content: center;

    transform: translateX(calc(var(--x) - 4px - 50%));
  }

  .pill {
    width: 2px;
    height: 100%;
    background-color: var(--secondary-text-color);
    opacity: 0.3;
    border-radius: 2px;
  }
</style>
