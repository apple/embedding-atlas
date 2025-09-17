<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type {
    DataField,
    DataPoint,
    EmbeddingViewConfig,
    Label,
    Point,
    Rectangle,
    ViewportState,
  } from "@embedding-atlas/component";
  import { EmbeddingViewMosaic } from "@embedding-atlas/component/svelte";
  import { Selection } from "@uwdata/mosaic-core";
  import { cubicOut } from "svelte/easing";

  import CategoryLegend from "./CategoryLegend.svelte";

  import type { Cache } from "./api.js";
  import { Context } from "./contexts.js";
  import type { EmbeddingLegend } from "./database_utils.js";
  import type { PlotStateStore } from "./plots/plot_state_store.js";
  import { interpolateViewport } from "./viewport_animation.js";

  const embeddingAnimationDuration = 800;

  interface Props {
    table: string;
    filter: Selection;

    id: string;
    x: string;
    y: string;
    text?: string | null;
    additionalFields?: Record<string, DataField>;
    categoryLegend?: EmbeddingLegend | null;

    config?: EmbeddingViewConfig | null;
    labels?: Label[] | null;

    customTooltip?: any;
    customOverlay?: any;

    onClickPoint?: ((point: DataPoint) => void) | null;

    stateStore?: PlotStateStore<{
      viewportState: ViewportState | null;
      rangeSelection: Rectangle | Point[] | null;
    } | null>;

    cache?: Cache | null;
  }

  let {
    table,
    filter,
    id,
    x,
    y,
    text,
    categoryLegend,
    additionalFields,
    config,
    labels,
    customTooltip,
    customOverlay,
    onClickPoint = null,
    stateStore,
    cache,
  }: Props = $props();

  const coordinator = Context.coordinator;
  const darkMode = Context.darkMode;

  let viewportState: ViewportState | null = $state.raw(null);
  let rangeSelectionValue: Rectangle | Point[] | null = $state.raw(null);
  let tooltip: any | null = $state.raw(null);
  let selection: any[] | null = $state.raw([]);

  let embeddingWidth = $state(750);
  let embeddingHeight = $state(750);

  let legendStateStore = $derived(stateStore?.child("legend"));

  // Viewport Animation

  let currentViewportAnimation: number | null;
  export function startViewportAnimation(newState: ViewportState) {
    tooltip = null;
    let start = viewportState;
    if (start == null) {
      viewportState = newState;
      return;
    }
    let duration = embeddingAnimationDuration;
    let t0 = new Date().getTime();
    let callback = () => {
      let t = (new Date().getTime() - t0) / duration;
      if (t > 1) {
        t = 1;
      } else {
        currentViewportAnimation = requestAnimationFrame(callback);
      }
      viewportState = interpolateViewport(start, newState, cubicOut(t));
    };
    if (currentViewportAnimation) {
      cancelAnimationFrame(currentViewportAnimation);
    }
    currentViewportAnimation = requestAnimationFrame(callback);
  }

  export function showTooltip(id: any | null) {
    if (id == null) {
      selection = [];
      tooltip = null;
    } else {
      selection = [id];
      tooltip = id;
    }
  }

  function clearViewportAnimation() {
    if (currentViewportAnimation) {
      cancelAnimationFrame(currentViewportAnimation);
    }
  }

  // Click point
  function emitClickPoint(point: DataPoint) {
    onClickPoint?.(point);
  }

  $effect(() => {
    if (selection?.[0] != null) {
      emitClickPoint(selection?.[0]);
    }
  });

  // Sync with state store
  $effect.pre(() => {
    let state = stateStore;
    if (!state) {
      return;
    }
    let cleanup = state.subscribe((s) => {
      if (s == null) {
        return;
      }
      viewportState = s.viewportState;
      rangeSelectionValue = s.rangeSelection;
    });
    $effect(() => {
      state.set({
        viewportState: viewportState,
        rangeSelection: rangeSelectionValue,
      });
    });
    return cleanup;
  });
</script>

<div
  class="absolute left-0 right-0 top-0 bottom-0"
  bind:clientWidth={embeddingWidth}
  bind:clientHeight={embeddingHeight}
>
  <EmbeddingViewMosaic
    coordinator={coordinator}
    table={table}
    identifier={id}
    x={x}
    y={y}
    labels={labels}
    config={{
      ...(config ?? {}),
      colorScheme: $darkMode ? "dark" : "light",
    }}
    text={text}
    category={categoryLegend?.indexColumn}
    categoryColors={categoryLegend?.legend.map((d) => d.color)}
    additionalFields={additionalFields}
    viewportState={viewportState}
    onViewportState={(v) => {
      viewportState = v;
      clearViewportAnimation();
    }}
    tooltip={tooltip}
    onTooltip={(v) => (tooltip = v)}
    selection={selection}
    onSelection={(v) => {
      selection = v;
    }}
    filter={filter}
    rangeSelection={filter}
    rangeSelectionValue={rangeSelectionValue}
    onRangeSelection={(v) => {
      rangeSelectionValue = v;
    }}
    width={embeddingWidth}
    height={embeddingHeight}
    customTooltip={customTooltip}
    customOverlay={customOverlay}
    cache={cache}
  />

  {#if categoryLegend != null}
    {#key categoryLegend}
      <CategoryLegend items={categoryLegend.legend} selection={filter} stateStore={legendStateStore} />
    {/key}
  {/if}
</div>
