<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { ChartViewProps } from "./chart.js";
  import { findChartComponent } from "./chart_types.js";

  let props: Omit<ChartViewProps, "width" | "height"> & {
    width?: number | "container";
    height?: number | "container";
  } = $props();
  let ComponentClass = $derived(findChartComponent(props.spec));

  let clientWidth = $state(100);
  let clientHeight = $state(100);

  function logError(node: HTMLElement, props: { spec: any; error: any }) {
    console.trace("Error happened in chart with spec", props.spec, props.error);
  }
</script>

<div
  style:width={props.width == "container" ? "100%" : props.width != undefined ? `${props.width}px` : "fit-content"}
  style:height={props.height == "container" ? "100%" : props.height != undefined ? `${props.height}px` : "fit-content"}
  bind:clientWidth={clientWidth}
  bind:clientHeight={clientHeight}
>
  <svelte:boundary>
    <ComponentClass
      {...props}
      width={props.width == "container" ? clientWidth : props.width}
      height={props.height == "container" ? clientHeight : props.height}
    />
    {#snippet failed(error, reset)}
      <button onclick={reset} use:logError={{ spec: props.spec, error: error }}>
        An occured with this chart. Click to retry.
      </button>
    {/snippet}
  </svelte:boundary>
</div>
