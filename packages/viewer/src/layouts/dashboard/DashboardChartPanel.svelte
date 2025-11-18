<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { interactionHandler, type CursorValue } from "@embedding-atlas/utils";
  import { fade } from "svelte/transition";

  import ConfirmButton from "../../charts/builder/ConfirmButton.svelte";
  import SpecEditor from "../../charts/builder/SpecEditor.svelte";
  import CornerButton from "../../widgets/CornerButton.svelte";

  import { IconCheck, IconClose, IconEdit } from "../../assets/icons.js";

  import type { ChartContext } from "../../charts/chart.js";
  import { findChartTypeOptions } from "../../charts/chart_types.js";
  import type { LayoutProps } from "../layout.js";
  import { type Grid } from "./placement.js";

  const RESIZER_OFFSET = -2;
  const RESIZER_SIZE = 10;

  interface Props {
    context: ChartContext;
    id: string;
    spec: any;

    order: number;

    grid: Grid;
    placement: { x: number; y: number; width: number; height: number };

    onRemove?: () => void;

    onBringToFront?: () => void;
    onIsPlacementChanging?: (value: boolean) => void;
    onPlacementChange?: (placement: { x: number; y: number; width: number; height: number }) => void;

    onSpecChange?: (spec: any) => void;

    chartView: LayoutProps<unknown>["chartView"];
  }

  let {
    context,
    id,
    spec,
    order,
    placement,
    grid,
    onBringToFront,
    onIsPlacementChanging,
    onPlacementChange,
    onRemove,
    onSpecChange,
    chartView,
  }: Props = $props();

  let { colorScheme } = context;

  let { x, y, width, height } = $derived(grid.resolvePlacement(placement));

  let isEditing = $state(false);
  let supportsEditMode = $derived(findChartTypeOptions(spec).supportsEditMode ?? false);
  let chartMode: "edit" | "view" = $derived(supportsEditMode && isEditing ? "edit" : "view");

  let editingSpec = $state.raw<any | undefined>();

  function dragHandler(mask: [number, number, number, number]) {
    return (e1: CursorValue) => {
      let numColumns = grid.numColumns;
      let xScaler = grid.xScaler;
      let yScaler = grid.yScaler;

      let px1 = placement.x;
      let py1 = placement.y;
      let px2 = placement.x + placement.width;
      let py2 = placement.y + placement.height;

      onBringToFront?.();
      onIsPlacementChanging?.(true);

      return {
        move: (e2: CursorValue) => {
          let nx1 = Math.round(px1 + ((e2.clientX - e1.clientX) * mask[0]) / xScaler);
          let ny1 = Math.round(py1 + ((e2.clientY - e1.clientY) * mask[1]) / yScaler);
          let nx2 = Math.round(px2 + ((e2.clientX - e1.clientX) * mask[2]) / xScaler);
          let ny2 = Math.round(py2 + ((e2.clientY - e1.clientY) * mask[3]) / yScaler);
          const minW = 2;
          const minH = 2;
          if (mask[0] && mask[2]) {
            nx1 = Math.min(Math.max(0, nx1), numColumns - (px2 - px1));
            nx2 = nx1 + (px2 - px1);
          } else {
            nx1 = Math.min(Math.max(0, nx1), px2 - minW);
            nx2 = Math.min(Math.max(px1 + minW, nx2), numColumns);
          }
          if (mask[1] && mask[3]) {
            ny1 = Math.max(0, ny1);
            ny2 = ny1 + (py2 - py1);
          } else {
            ny1 = Math.min(Math.max(0, ny1), py2 - minH);
            ny2 = Math.max(py1 + minH, ny2);
          }
          onPlacementChange?.({
            x: nx1,
            y: ny1,
            width: nx2 - nx1,
            height: ny2 - ny1,
          });
        },
        up: () => {
          onIsPlacementChanging?.(false);
        },
        cancel: () => {
          onIsPlacementChanging?.(false);
        },
      };
    };
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div
  class="absolute"
  style:left="{x}px"
  style:top="{y}px"
  style:width="{width}px"
  style:height="{height}px"
  style:z-index={order}
  transition:fade={{ duration: 100 }}
  onclick={() => {
    onBringToFront?.();
  }}
>
  <div
    class="absolute left-0 right-0 top-0 bottom-0 bg-white dark:bg-black rounded-md overflow-hidden flex flex-col shadow-md border border-slate-300 dark:border-slate-700 group"
  >
    <div class="p-2 bg-slate-100 dark:bg-slate-900 flex cursor-move select-none">
      <div
        class="flex-1 font-mono font-medium whitespace-nowrap overflow-hidden text-ellipsis"
        use:interactionHandler={{
          click: () => {
            onBringToFront?.();
          },
          drag: dragHandler([1, 1, 1, 1]),
        }}
      >
        {spec.title}
      </div>
      <div class="flex gap-1 sm:opacity-0 group-hover:opacity-100">
        <CornerButton
          icon={chartMode == "edit" ? IconCheck : IconEdit}
          onClick={() => {
            isEditing = !isEditing;
            if (isEditing) {
              editingSpec = spec;
            }
          }}
        />
        <CornerButton icon={IconClose} onClick={onRemove} />
      </div>
    </div>
    <div class="flex-1 min-h-0">
      {#if !(chartMode == "view" && isEditing && onSpecChange)}
        {@render chartView({ id: id, width: "container", height: "container", mode: chartMode })}
      {:else}
        <div class="p-2 h-full flex flex-col gap-2">
          <SpecEditor
            class="w-full flex-1 min-h-0"
            initialValue={editingSpec}
            colorScheme={$colorScheme}
            onChange={(s) => {
              editingSpec = s;
            }}
          />
          <ConfirmButton
            label="Confirm"
            disabled={editingSpec == undefined}
            onClick={() => {
              if (editingSpec != undefined) {
                onSpecChange(editingSpec);
                isEditing = false;
              }
            }}
          />
        </div>
      {/if}
    </div>
  </div>
  <!-- Resize and move controls -->
  <div
    style:left="{RESIZER_OFFSET}px"
    style:top="{RESIZER_SIZE}px"
    style:bottom="{RESIZER_SIZE}px"
    style:width="{RESIZER_SIZE}px"
    class="absolute cursor-ew-resize"
    use:interactionHandler={{
      drag: dragHandler([1, 0, 0, 0]),
    }}
  ></div>
  <div
    style:right="{RESIZER_OFFSET}px"
    style:top="{RESIZER_SIZE}px"
    style:bottom="{RESIZER_SIZE}px"
    style:width="{RESIZER_SIZE}px"
    class="absolute cursor-ew-resize"
    use:interactionHandler={{
      drag: dragHandler([0, 0, 1, 0]),
    }}
  ></div>
  <div
    style:left="{RESIZER_SIZE}px"
    style:right="{RESIZER_SIZE}px"
    style:top="{RESIZER_OFFSET}px"
    style:height="{RESIZER_SIZE}px"
    class="absolute cursor-ns-resize"
    use:interactionHandler={{
      drag: dragHandler([0, 1, 0, 0]),
    }}
  ></div>
  <div
    style:left="{RESIZER_SIZE}px"
    style:right="{RESIZER_SIZE}px"
    style:bottom="{RESIZER_OFFSET}px"
    style:height="{RESIZER_SIZE}px"
    class="absolute cursor-ns-resize"
    use:interactionHandler={{
      drag: dragHandler([0, 0, 0, 1]),
    }}
  ></div>
  <div
    style:left="{RESIZER_OFFSET}px"
    style:top="{RESIZER_OFFSET}px"
    style:width="{RESIZER_SIZE}px"
    style:height="{RESIZER_SIZE}px"
    class="absolute cursor-nwse-resize"
    use:interactionHandler={{
      drag: dragHandler([1, 1, 0, 0]),
    }}
  ></div>
  <div
    style:right="{RESIZER_OFFSET}px"
    style:top="{RESIZER_OFFSET}px"
    style:width="{RESIZER_SIZE}px"
    style:height="{RESIZER_SIZE}px"
    class="absolute cursor-nesw-resize"
    use:interactionHandler={{
      drag: dragHandler([0, 1, 1, 0]),
    }}
  ></div>
  <div
    style:left="{RESIZER_OFFSET}px"
    style:bottom="{RESIZER_OFFSET}px"
    style:width="{RESIZER_SIZE}px"
    style:height="{RESIZER_SIZE}px"
    class="absolute cursor-nesw-resize"
    use:interactionHandler={{
      drag: dragHandler([1, 0, 0, 1]),
    }}
  ></div>
  <div
    style:right="{RESIZER_OFFSET}px"
    style:bottom="{RESIZER_OFFSET}px"
    style:width="{RESIZER_SIZE}px"
    style:height="{RESIZER_SIZE}px"
    class="absolute cursor-nwse-resize"
    use:interactionHandler={{
      drag: dragHandler([0, 0, 1, 1]),
    }}
  ></div>
</div>
