<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { get } from "svelte/store";

  import { screenshot } from "../utils/screenshot.js";
  import type { ChartViewProps } from "./chart.js";
  import { findChartComponent } from "./chart_types.js";

  let props: Omit<ChartViewProps<any, any>, "width" | "height"> & {
    width?: number | "container";
    height?: number | "container";
  } = $props();

  let clientWidth = $state(100);
  let clientHeight = $state(100);

  let { spec, width, height } = $derived(props);
  let chartState = $derived(props.state ?? {});

  let styleWidth = $derived(width == "container" ? "100%" : width != undefined ? `${width}px` : "fit-content");
  let styleHeight = $derived(height == "container" ? "100%" : height != undefined ? `${height}px` : "fit-content");
  let componentWidth = $derived(width == "container" ? clientWidth : width);
  let componentHeight = $derived(height == "container" ? clientHeight : height);

  let ComponentClass = $derived(findChartComponent(spec));

  let container: HTMLDivElement;

  function logError(_: HTMLElement, props: { spec: any; error: any }) {
    console.trace("Error happened in chart with spec", props.spec, props.error);
  }

  $effect(() =>
    props.registerDelegate?.({
      screenshot: async (options) => {
        let colorScheme = get(props.context.colorScheme);
        return await screenshot(container, {
          ...options,
          backgroundColor: colorScheme == "dark" ? "#000000" : "#ffffff",
        });
      },
    }),
  );
</script>

<div
  style:width={styleWidth}
  style:height={styleHeight}
  class="bg-white dark:bg-black"
  bind:clientWidth={clientWidth}
  bind:clientHeight={clientHeight}
  bind:this={container}
>
  <svelte:boundary>
    <ComponentClass {...props} state={chartState} width={componentWidth} height={componentHeight} />
    {#snippet failed(error, reset)}
      <div class="flex w-full h-full justify-center items-center">
        <button
          class="flex gap-1 items-center p-4 text-slate-400 dark:text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
          onclick={reset}
          use:logError={{ spec: spec, error: error }}
        >
          An error occured with this chart. Click to retry.
        </button>
      </div>
    {/snippet}
  </svelte:boundary>
</div>
