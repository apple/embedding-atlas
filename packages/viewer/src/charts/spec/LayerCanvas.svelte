<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  import type { ChartTheme } from "../common/theme.js";
  import type { XYFrameProxy } from "../common/types.js";
  import {
    areaPath,
    lineData,
    linePath,
    resolveBand,
    resolveCanvasStyle,
    resolvePoint,
    resolveSize,
    type ResolveContext,
  } from "./layer_helper.js";
  import type { LayerOutputs } from "./runtime.js";

  function draw(ctx: CanvasRenderingContext2D, rctx: ResolveContext, layer: LayerOutputs) {
    switch (layer.primitive) {
      case "rect": {
        const x = resolveBand(rctx, "x", layer.xDimension);
        const y = resolveBand(rctx, "y", layer.yDimension);
        let style = resolveCanvasStyle(rctx, layer.style);
        style.prepare(ctx);
        for (let i = 0; i < layer.data.length; i++) {
          const [x0, x1] = x.at(i);
          const [y0, y1] = y.at(i);
          if (isFinite(x0) && isFinite(x1) && isFinite(y0) && isFinite(y1)) {
            let rx = Math.min(x0, x1);
            let ry = Math.min(y0, y1);
            let rw = Math.abs(x1 - x0);
            let rh = Math.abs(y1 - y0);
            ctx.beginPath();
            ctx.rect(rx, ry, rw, rh);
            style.draw(ctx, i);
          }
        }
        break;
      }
      case "point": {
        const x = resolvePoint(rctx, "x");
        const y = resolvePoint(rctx, "y");
        const size = resolveSize(rctx, layer.pointSize);
        let style = resolveCanvasStyle(rctx, layer.style);
        style.prepare(ctx);
        for (let i = 0; i < layer.data.length; i++) {
          let xi = x.at(i);
          let yi = y.at(i);
          let sizei = size.at(i);
          if (isFinite(xi) && isFinite(yi) && isFinite(sizei)) {
            ctx.beginPath();
            ctx.arc(xi, yi, Math.sqrt(sizei / Math.PI), 0, Math.PI * 2);
            style.draw(ctx, i);
          }
        }
        break;
      }
      case "rule": {
        const x = resolveBand(rctx, "x", layer.xDimension);
        const y = resolveBand(rctx, "y", layer.yDimension);
        let style = resolveCanvasStyle(rctx, layer.style);
        style.prepare(ctx);
        for (let i = 0; i < layer.data.length; i++) {
          const [x0, x1] = x.at(i);
          const [y0, y1] = y.at(i);
          if (isFinite(x0) && isFinite(x1) && isFinite(y0) && isFinite(y1)) {
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            style.draw(ctx, i);
          }
        }
        break;
      }
      case "line": {
        let style = resolveCanvasStyle(rctx, layer.style);
        style.prepare(ctx);
        for (let ids of lineData(layer.data)) {
          ctx.beginPath();
          linePath(rctx, ids, layer, ctx);
          style.draw(ctx, ids[0]);
        }
        break;
      }
      case "area": {
        let style = resolveCanvasStyle(rctx, layer.style);
        style.prepare(ctx);
        for (let ids of lineData(layer.data)) {
          ctx.beginPath();
          areaPath(rctx, ids, layer, ctx);
          style.draw(ctx, ids[0]);
        }
        break;
      }
    }
  }

  function drawCanvas(
    canvas: HTMLCanvasElement,
    margin: number,
    pixelRatio: number,
    rctx: ResolveContext,
    layer: LayerOutputs,
  ) {
    let ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    ctx.reset();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(pixelRatio, pixelRatio);
    ctx.translate(margin, margin);

    draw(ctx, rctx, layer);
  }
</script>

<script lang="ts">
  interface Props {
    proxy: XYFrameProxy;
    theme: ChartTheme;
    layer: LayerOutputs;
  }

  let { proxy, theme, layer }: Props = $props();
  let width = $derived(proxy.plotWidth);
  let height = $derived(proxy.plotHeight);
  let margin = $state(24);
  let pixelRatio = $state(2);
  let canvas: HTMLCanvasElement;

  $effect(() => {
    drawCanvas(canvas, margin, pixelRatio, { proxy, data: layer.data, theme }, layer);
  });
</script>

<foreignObject x={-margin} y={-margin} width={width + margin * 2} height={height + margin * 2}>
  <canvas
    bind:this={canvas}
    width={(width + margin * 2) * pixelRatio}
    height={(height + margin * 2) * pixelRatio}
    style:width="{width + margin * 2}px"
    style:height="{height + margin * 2}px"
  ></canvas>
</foreignObject>
