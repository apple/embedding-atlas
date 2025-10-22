<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { DataPoint, OverlayProxy } from "@embedding-atlas/component";

  interface Props {
    center?: DataPoint | null;
    points?: DataPoint[];
    proxy: OverlayProxy;
  }

  let { center, points, proxy }: Props = $props();
</script>

<svg width={proxy.width} height={proxy.height}>
  <g>
    {#if center != null}
      {@const l1 = proxy.location(center.x, center.y)}
      {#each points ?? [] as point}
        {@const l2 = proxy.location(point.x, point.y)}
        <line x1={l1.x} y1={l1.y} x2={l2.x} y2={l2.y} class="stroke-orange-500" />
      {/each}
    {/if}
    {#each points ?? [] as point}
      {@const loc = proxy.location(point.x, point.y)}
      <circle cx={loc.x} cy={loc.y} r={4} class="fill-orange-500 stroke-orange-700 stroke-2" />
    {/each}
  </g>
</svg>
