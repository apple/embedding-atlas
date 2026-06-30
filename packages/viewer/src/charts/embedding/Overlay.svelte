<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { DataPoint, OverlayProxy } from "@embedding-atlas/component";

  interface Props {
    nodes?: DataPoint[];
    edges?: { start: DataPoint; end: DataPoint }[];
    proxy: OverlayProxy;
  }

  let { nodes, edges, proxy }: Props = $props();
</script>

<svg width={proxy.width} height={proxy.height}>
  <g>
    {#each edges ?? [] as e}
      {@const p1 = proxy.location(e.start.x, e.start.y, e.start.z)}
      {@const p2 = proxy.location(e.end.x, e.end.y, e.end.z)}
      {#if isFinite(p1.x) && isFinite(p1.y) && isFinite(p2.x) && isFinite(p2.y)}
        <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} class="stroke-orange-500" />
      {/if}
    {/each}
  </g>
  <g>
    {#each nodes ?? [] as p}
      {@const loc = proxy.location(p.x, p.y, p.z)}
      {#if isFinite(loc.x) && isFinite(loc.y)}
        <circle cx={loc.x} cy={loc.y} r={4} class="fill-orange-500 stroke-orange-700 stroke-2" />
      {/if}
    {/each}
  </g>
</svg>
