<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { rgb } from "d3-color";
  import type { XYFrameProxy } from "./types.js";

  interface Props {
    color: (x: number, y: number) => string;
    rasterWidth: number;
    rasterHeight: number;

    proxy: XYFrameProxy;

    xDomain?: [number, number];
    yDomain?: [number, number];
  }

  let { proxy, xDomain, yDomain, color, rasterWidth, rasterHeight }: Props = $props();

  let x0 = $derived(proxy.xScale && xDomain ? proxy.xScale.apply(xDomain[0]) : 0);
  let x1 = $derived(proxy.xScale && xDomain ? proxy.xScale.apply(xDomain[1]) : proxy.plotWidth);

  let y0 = $derived(proxy.yScale && yDomain ? proxy.yScale.apply(yDomain[0]) : 0);
  let y1 = $derived(proxy.yScale && yDomain ? proxy.yScale.apply(yDomain[1]) : proxy.plotHeight);

  let href: string | null = $state(null);

  let canvas = document.createElement("canvas");

  $effect.pre(() => {
    canvas.width = rasterWidth;
    canvas.height = rasterHeight;
    let ctx = canvas.getContext("2d", { colorSpace: "srgb", willReadFrequently: true })!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let data = ctx.getImageData(0, 0, canvas.width, canvas.height);

    let offset = 0;
    for (let py = 0; py < canvas.height; py++) {
      for (let px = 0; px < canvas.width; px++) {
        let cx = ((px + 0.5) / canvas.width) * (x1 - x0) + x0;
        let cy = ((py + 0.5) / canvas.height) * (y1 - y0) + y0;
        let x = proxy.xScale?.invert(cx) ?? 0;
        let y = proxy.yScale?.invert(cy) ?? 0;
        let { r, g, b, opacity } = rgb(color(x, y));
        data.data[offset++] = r;
        data.data[offset++] = g;
        data.data[offset++] = b;
        data.data[offset++] = opacity * 255;
      }
    }

    ctx.putImageData(data, 0, 0);
    href = canvas.toDataURL("image/png");
  });
</script>

<image
  x={Math.min(x0, x1)}
  y={Math.min(y0, y1)}
  width={Math.abs(x0 - x1)}
  height={Math.abs(y0 - y1)}
  href={href}
  preserveAspectRatio="none"
/>
