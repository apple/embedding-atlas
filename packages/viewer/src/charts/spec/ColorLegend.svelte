<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Raster from "../common/Raster.svelte";
  import SizeReader from "../common/SizeReader.svelte";
  import XYFrame from "../common/XYFrame.svelte";

  import type { ChartTheme } from "../common/theme.js";
  import type { ConcreteScale } from "../common/types.js";
  import type { ChartOutputs } from "./runtime.js";

  interface Props {
    scale: ConcreteScale<string>;
    outputs: ChartOutputs;
    theme: ChartTheme;
  }

  let { scale, outputs, theme }: Props = $props();
</script>

<div class="text-sm text-slate-400 dark:text-slate-500">
  {#if scale.type == "band"}
    <div class="flex gap-2 flex-wrap items-center select-none">
      {#each scale.domain.concat(scale.specialValues ?? []) as item}
        {@const color = scale.apply(item)}
        <div class="flex gap-1 items-center" title={item}>
          <div class="w-3 h-3 block rounded-sm" style:background={color}></div>
          <div class="whitespace-nowrap max-w-32 overflow-hidden text-ellipsis">
            {item}
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="pl-4">
      <SizeReader>
        {#snippet children(width, _)}
          <XYFrame
            xScale={{
              type: outputs.scale.color.type ?? "linear",
              domain: scale.domain,
              specialValues: scale.specialValues,
            }}
            xAxis={{ extendScaleToTicks: false }}
            width={width}
            plotHeight={8}
            theme={theme}
          >
            {#snippet children(proxy)}
              <Raster
                color={(v) => scale.apply(v)}
                rasterWidth={100}
                rasterHeight={1}
                proxy={proxy}
                xDomain={proxy.scale.x?.domain as any}
              />
              {#each scale.specialValues ?? [] as v}
                {@const [x0, x1] = proxy.scale.x?.applyBand(v) ?? [NaN, NaN]}
                <rect
                  x={Math.min(x0, x1)}
                  width={Math.abs(x1 - x0)}
                  y={0}
                  height={proxy.plotHeight}
                  fill={scale.apply(v)}
                />
              {/each}
            {/snippet}
          </XYFrame>
        {/snippet}
      </SizeReader>
    </div>
  {/if}
</div>
