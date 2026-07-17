<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { ListItem } from "./features_list_store.js";

  interface Props {
    item: ListItem;
    /** Reference value (max across the list) for width scaling. */
    maxCount: number;
    /** Pixel width of the bar. */
    width: number;
    /** Pixel height of the bar. Default 8. */
    height?: number;
    /** When true, `item.sourceSegments` is rendered as colored segments
     *  on top of a faded total bar. When false, a plain count bar is used. */
    segmented?: boolean;
    /** Color per source. Falls back to `markColor`. Used only when `segmented`. */
    segmentColors?: Record<string, string>;
    /** Color of the foreground (selected) bar. */
    markColor: string;
    /** Color of the faded background (unselected total) bar. */
    markColorFade: string;
  }

  let {
    item,
    maxCount,
    width,
    height = 10,
    segmented = false,
    segmentColors = {},
    markColor,
    markColorFade,
  }: Props = $props();

  let useSegments = $derived(segmented && item.sourceSegments != null && item.sourceSegments.length > 0);
</script>

{#if useSegments}
  {@const segs = item.sourceSegments!}
  {@const totalW = (item.count / Math.max(maxCount, 1)) * width}
  <div class="relative flex-none" style:width="{width}px" style:height="{height}px">
    <!-- Wrapper sized to totalW so rounded-sm/overflow-hidden clip at the bar's
         actual right edge. Acts as the faded total background. -->
    <div
      class="absolute left-0 top-0 bottom-0 rounded-sm overflow-hidden"
      style:width="{Math.max(totalW, item.count > 0 ? 0.5 : 0)}px"
      style:background={markColorFade}
    >
      <!-- Colored selected segments stacked left-to-right by countSelected -->
      {#each segs as seg, i (seg.source)}
        {@const offset = segs.slice(0, i).reduce((a, b) => a + b.countSelected, 0)}
        {@const segW = (seg.countSelected / Math.max(maxCount, 1)) * width}
        <div
          class="absolute top-0 bottom-0"
          style:left="{(offset / Math.max(maxCount, 1)) * width}px"
          style:width="{Math.max(segW, seg.countSelected > 0 ? 0.5 : 0)}px"
          style:background={segmentColors[seg.source] ?? markColor}
          title={seg.count !== seg.countSelected
            ? `${seg.source}: ${seg.countSelected.toLocaleString()} / ${seg.count.toLocaleString()}`
            : `${seg.source}: ${seg.count.toLocaleString()}`}
        ></div>
      {/each}
    </div>
  </div>
{:else}
  {@const total = item.count}
  {@const sel = item.countSelected}
  {@const totalW = (total / Math.max(maxCount, 1)) * width}
  {@const selW = (sel / Math.max(maxCount, 1)) * width}
  <div class="relative flex-none" style:width="{width}px" style:height="{height}px">
    <div
      class="absolute left-0 top-0 bottom-0 rounded-sm overflow-hidden"
      style:width="{Math.max(totalW, total > 0 ? 0.5 : 0)}px"
      style:background={markColorFade}
    >
      <div
        class="absolute left-0 top-0 bottom-0"
        style:width="{Math.max(selW, sel > 0 ? 0.5 : 0)}px"
        style:background={markColor}
      ></div>
    </div>
  </div>
{/if}
