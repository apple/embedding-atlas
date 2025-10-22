<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { polygonToPath, type Point } from "../utils.js";

  interface Props {
    value: Point[];
    pointLocation: (x: number, y: number) => Point;
  }

  let { value, pointLocation }: Props = $props();

  let locs = $derived(value.map(({ x, y }) => pointLocation(x, y)));
</script>

<path d={polygonToPath(locs)} style:stroke="#fff" style:fill="rgba(128,128,128,0.25)" />
