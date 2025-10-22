<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { LayoutProps } from "./layout.js";

  let { charts, chartView }: LayoutProps<{}> = $props();

  let containerWidth = $state(100);
  let containerHeight = $state(100);

  let allIDs = $derived(Object.keys(charts));

  let chartWidth = $derived(containerWidth / 3 - 10);
  let chartHeight = 300;
</script>

<div
  class="w-full h-full flex flex-row flex-wrap gap-2 overflow-y-scroll"
  bind:clientWidth={containerWidth}
  bind:clientHeight={containerHeight}
>
  {#each allIDs as id (id)}
    <div class="bg-white dark:bg-black rounded-md flex flex-col overflow-hidden" style:width="{chartWidth}px">
      <div class="p-2 bg-slate-100">{charts[id].title}</div>
      {@render chartView({
        id: id,
        width: chartWidth,
        height: chartHeight,
      })}
    </div>
  {/each}
</div>
