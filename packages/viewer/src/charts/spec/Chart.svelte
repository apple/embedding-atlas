<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onDestroy } from "svelte";

  import Container from "../common/Container.svelte";
  import SizeReader from "../common/SizeReader.svelte";
  import XYFrame from "../common/XYFrame.svelte";
  import XYSelection from "../common/XYSelection.svelte";
  import ColorLegend from "./ColorLegend.svelte";
  import Layer from "./Layer.svelte";
  import Widgets from "./Widgets.svelte";

  import type { ChartViewProps } from "../chart.js";
  import { inferColorScale, inferSizeScale } from "../common/infer.js";
  import { resolveChartTheme } from "../common/theme.js";
  import { ChartRuntime } from "./runtime.js";
  import type { ChartSpec } from "./spec.js";

  interface ChartState {
    [key: string]: any;
  }

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onStateChange,
    onSpecChange,
  }: ChartViewProps<ChartSpec, ChartState> = $props();

  let { colorScheme, theme: themeConfig } = context;

  let theme = $derived(resolveChartTheme($colorScheme, $themeConfig));

  let runtime = new ChartRuntime(context, () => {
    onStateChange({}, "replace");
  });
  let { outputs, selections } = runtime;

  onDestroy(() => {
    runtime.destroy();
  });

  $effect.pre(() => {
    runtime.setSpec(spec);
  });

  $effect.pre(() => {
    runtime.state.set(chartState);
  });

  // Non-position scales
  let colorScale = $derived($outputs?.scale.color ? inferColorScale($outputs.scale.color, theme) : undefined);
  let sizeScale = $derived($outputs?.scale.size ? inferSizeScale($outputs.scale.size, theme) : undefined);

  let zOrderedLayers = $derived($outputs?.layers.sort((a, b) => a.zIndex - b.zIndex) ?? []);

  let flexHeight = $derived(height != null);
  let defaultPlotHeight = 160;
</script>

<Container width={width} height={height} class="flex flex-col gap-2">
  <div class="flex-1 flex flex-col">
    {#if $outputs?.axis.y.title != undefined}
      <div class="text-slate-400 mb-1 select-none truncate">↑ {$outputs?.axis.y.title}</div>
    {/if}
    <SizeReader flexHeight={flexHeight}>
      {#snippet children(parentWidth, parentHeight)}
        {#if $outputs != undefined}
          <XYFrame
            width={parentWidth}
            height={flexHeight ? parentHeight : undefined}
            plotWidth={spec.plotSize?.width ?? undefined}
            plotHeight={spec.plotSize?.height ??
              (spec.plotSize?.aspectRatio == undefined && !flexHeight ? defaultPlotHeight : undefined)}
            plotAspectRatio={spec.plotSize?.aspectRatio}
            xScale={$outputs.scale.x}
            yScale={$outputs.scale.y}
            xAxis={$outputs.axis.x}
            yAxis={$outputs.axis.y}
            otherScales={{
              color: colorScale,
              size: sizeScale,
            }}
            theme={theme}
          >
            {#snippet childrenBelow(proxy)}
              {#each zOrderedLayers.filter((x) => x.zIndex < 0) as layer (layer.key)}
                <Layer proxy={proxy} theme={theme} layer={layer} />
              {/each}
            {/snippet}
            {#snippet children(proxy)}
              {#each zOrderedLayers.filter((x) => !(x.zIndex < 0)) as layer (layer.key)}
                <Layer proxy={proxy} theme={theme} layer={layer} />
              {/each}

              {#each $selections ?? [] as selection}
                <XYSelection
                  proxy={proxy}
                  mode={selection.type}
                  value={chartState[selection.key]}
                  theme={theme}
                  onChange={(v) => {
                    switch (selection.type) {
                      case "x":
                        onStateChange({
                          [selection.key]: v?.x != undefined ? { x: v.x } : undefined,
                        });
                        break;
                      case "y":
                        onStateChange({
                          [selection.key]: v?.y != undefined ? { y: v.y } : undefined,
                        });
                        break;
                      case "xy":
                        onStateChange({
                          [selection.key]:
                            v != undefined && v.x != undefined && v.y != undefined ? { x: v.x, y: v.y } : undefined,
                        });
                        break;
                    }
                  }}
                />
              {/each}
            {/snippet}
          </XYFrame>
        {/if}
      {/snippet}
    </SizeReader>
    {#if $outputs?.axis.x.title != undefined}
      <div class="text-slate-400 mb-1 select-none text-right truncate">
        {$outputs?.axis.x.title} →
      </div>
    {/if}
  </div>

  {@const hasLegend = colorScale != undefined}
  {@const hasWidgets = (spec.widgets?.length ?? 0) > 0}

  <!-- Bottom section for legend and widgets -->
  {#if $outputs && (hasLegend || hasWidgets)}
    <div class="flex-none flex items-start gap-4 justify-end">
      {#if hasLegend}
        <div class="flex-1">
          <div class="min-w-32 max-w-72">
            <ColorLegend scale={colorScale} outputs={$outputs} theme={theme} />
          </div>
        </div>
      {/if}
      {#if hasWidgets}
        <Widgets spec={spec} outputs={$outputs} onSpecChange={onSpecChange} />
      {/if}
    </div>
  {/if}
</Container>
